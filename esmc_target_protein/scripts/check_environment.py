#!/usr/bin/env python
"""Environment preflight. Needs no dataset. Writes ``preflight_report.json``.

Exits non-zero on a fatal problem so a real GPU run does not start and then
fail half-way:

  * Python / PyTorch / Transformers versions and whether ``EsmcModel`` imports;
  * CUDA availability, GPU name and memory;
  * Hugging Face Hub reachability;
  * resolution AND verification of every checkpoint candidate;
  * tokenizer + model load;
  * one forward pass on "ACDEFGHIKL" with the Section 5 startup checks
    (tokens, residue alignment, hidden-state length) and the Section 6
    X/B/Z/J/U/O support tests through the complete tokenizer->model->extraction
    path;
  * the derived residue capacity.

Never falls back to another model or to another device.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from esmc_target.config import load_config  # noqa: E402
from esmc_target.errors import FatalError  # noqa: E402
from esmc_target.utils import (  # noqa: E402
    atomic_write_json,
    gpu_info,
    human_bytes,
    local_now_iso,
    setup_logging,
    software_versions,
)


def hub_reachable(timeout: int = 20) -> dict:
    try:
        import huggingface_hub
        client = huggingface_hub.HfApi()
        client.list_models(search="ESMC-600M", limit=3)
        return {"reachable": True, "detail": "huggingface_hub API call succeeded"}
    except Exception as exc:
        return {"reachable": False, "detail": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preflight check for a real ESMC run.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "preflight_report.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-forward", action="store_true",
                        help="Stop after checkpoint resolution (no tokenizer/model forward pass)")
    args = parser.parse_args(argv)
    logger = setup_logging(None)

    report: dict = {"timestamp": local_now_iso(), "fatal": False, "fatal_reasons": []}
    config = load_config(args.config)
    report["software_versions"] = software_versions().to_dict()
    info = gpu_info()
    report["gpu"] = info
    report["gpu"]["gpu_memory_total_human"] = human_bytes(info.get("gpu_memory_total_bytes"))
    logger.info("Python %s | PyTorch %s | Transformers %s", report["software_versions"]["python"],
                report["software_versions"]["pytorch"], report["software_versions"]["transformers"])
    logger.info("CUDA available: %s | %s | %s", info["cuda_available"], info["gpu_name"],
                report["gpu"]["gpu_memory_total_human"])
    if not info["cuda_available"]:
        logger.warning("No CUDA device: the pipeline still works on CPU but is slow and resumable.")

    try:
        from transformers import EsmcModel  # noqa: F401
        report["esmc_native_support"] = True
        logger.info("Native Transformers ESMC support: EsmcModel imports OK")
    except Exception as exc:
        report["esmc_native_support"] = False
        report["esmc_import_error"] = f"{type(exc).__name__}: {exc}"
        report["fatal"] = True
        report["fatal_reasons"].append(
            f"transformers has no native ESMC support ({exc}). Install with "
            f"'pip install -U transformers', else 'pip install git+https://github.com/"
            f"huggingface/transformers.git'.")
        logger.error("Native ESMC support MISSING: %s", exc)

    report["hub"] = hub_reachable()
    logger.info("Hugging Face Hub reachable: %s (%s)", report["hub"]["reachable"],
                report["hub"]["detail"])
    if not report["hub"]["reachable"] and not args.local_files_only:
        report["fatal"] = True
        report["fatal_reasons"].append(
            "huggingface.co is not reachable and local_files_only is false. The ESMC checkpoint "
            "cannot be downloaded. Use a machine with network access, or pre-populate the HF cache "
            "and pass --local-files-only.")
        logger.error("Hugging Face Hub is NOT reachable: %s", report["hub"]["detail"])

    resolved_payload = None
    if report.get("esmc_native_support"):
        from esmc_target.esmc_encoder import (  # noqa: WPS433
            DTYPE_MAP,
            inspect_runtime,
            probe_special_residue_support,
            resolve_model,
        )
        try:
            resolved = resolve_model(config.model.candidates,
                                     trust_remote_code=config.model.trust_remote_code,
                                     cache_dir=config.model.cache_dir,
                                     local_files_only=args.local_files_only)
            resolved_payload = resolved.to_dict()
            logger.info("Checkpoint verified: %s @ %s", resolved.model_id, resolved.model_revision)
        except FatalError as exc:
            report["fatal"] = True
            report["fatal_reasons"].append(str(exc))
            report["candidate_outcomes"] = [c for c in getattr(exc, "outcomes", [])]
            logger.error("Checkpoint resolution FAILED: %s", exc)
        except Exception as exc:  # pragma: no cover
            report["fatal"] = True
            report["fatal_reasons"].append(f"{type(exc).__name__}: {exc}")
            logger.error("Checkpoint resolution crashed: %s\n%s", exc, traceback.format_exc())
    report["resolved_model"] = resolved_payload

    if resolved_payload and not args.skip_forward:
        from esmc_target.esmc_encoder import DTYPE_MAP, inspect_runtime, probe_special_residue_support
        device = torch.device("cuda" if (args.device in (None, "auto", "cuda") and torch.cuda.is_available())
                              else "cpu")
        try:
            resolved = resolve_model(config.model.candidates,
                                     trust_remote_code=config.model.trust_remote_code,
                                     cache_dir=config.model.cache_dir,
                                     local_files_only=args.local_files_only)
            inspection = inspect_runtime(resolved.model, resolved.tokenizer, device=device,
                                         dtype=DTYPE_MAP[config.runtime.dtype])
            report["startup_inspection"] = inspection.to_dict()
            support = probe_special_residue_support(resolved.model, resolved.tokenizer,
                                                    device=device,
                                                    dtype=DTYPE_MAP[config.runtime.dtype])
            report["special_residue_support"] = {s: r.to_dict() for s, r in support.items()}
            logger.info("Residue capacity: max_positions=%d special_tokens=%d capacity=%d",
                        inspection.model_max_positions, inspection.special_tokens,
                        inspection.residue_capacity)
            for symbol, result in support.items():
                logger.info("  support %s -> %s (%s)", symbol, result.supported, result.detail)

            probe_lines = [
                "================================================================",
                "SPECIAL RESIDUE SUPPORT (X B Z J U O) — tokenizer/model probe",
                "================================================================",
            ]
            for symbol in ("X", "B", "Z", "J", "U", "O"):
                if symbol in support:
                    r = support[symbol]
                    st = "supported" if r.supported else "unsupported"
                    extra = "   <-- affects NSC-92859 / TXNRD1" if symbol == "U" else ""
                    probe_lines.append(f"  {symbol}: {st} ({r.detail}){extra}")
                else:
                    probe_lines.append(f"  {symbol}: unprobed")
            policy = config.sequence.special_residue_policy
            probe_lines.append(f"Current special_residue_policy: {policy}")
            u_ok = support.get("U").supported if "U" in support else False
            if not u_ok:
                probe_lines.append("  -> if U is UNSUPPORTED: the TXNRD1 row (NSC-92859) will FAIL under this policy.")
                probe_lines.append("  -> to allow it through as an approximation instead, rerun with")
                probe_lines.append("     --special-residue-policy replace_with_X (changes the residue's biological")
                probe_lines.append("     identity; see README).")
            else:
                probe_lines.append("  -> U is SUPPORTED: TXNRD1 row (NSC-92859) will be processed natively.")
            probe_lines.append("================================================================")
            summary_text = "\n".join(probe_lines)
            report["special_residue_probe_summary"] = summary_text
            logger.info("\n%s", summary_text)
        except Exception as exc:  # pragma: no cover
            report["fatal"] = True
            report["fatal_reasons"].append(f"forward/self-check failed: {type(exc).__name__}: {exc}")
            logger.error("Forward/self-check failed: %s\n%s", exc, traceback.format_exc())

    report["fatal"] = bool(report["fatal"])
    out = Path(args.out)
    atomic_write_json(out, report)
    logger.info("Wrote %s", out)
    if report["fatal"]:
        logger.error("PREFLIGHT FAILED")
        for reason in report["fatal_reasons"]:
            logger.error("  - %s", reason)
        return 1
    logger.info("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
