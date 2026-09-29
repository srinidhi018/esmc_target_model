#!/usr/bin/env python
"""Run the offline ESMC-600M target-protein embedding pipeline.

Examples
--------
Smoke test (isolated output dir, first five rows only)::

    python scripts/run_preprocessing.py --input data/targetprotein.csv \
        --config configs/config.yaml --max-rows 5 --output-dir outputs/smoke_test

Full run, then resume (computes 0 new proteins)::

    python scripts/run_preprocessing.py --input data/targetprotein.csv \
        --config configs/config.yaml --output-dir outputs/full --resume
    python scripts/validate_outputs.py --output-dir outputs/full
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esmc_target.config import load_config  # noqa: E402
from esmc_target.errors import FatalError, IsolationError, PipelineInterruptedError  # noqa: E402
from esmc_target.pipeline import PreprocessingPipeline, enforce_run_isolation  # noqa: E402
from esmc_target.utils import get_logger, setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline ESMC-600M target-protein preprocessing (label-free, frozen model).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to targetprotein.csv")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "config.yaml"),
                        help="YAML config file")
    parser.add_argument("--output-dir", default=None, help="Output directory for all artifacts")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None,
                        help="Inference dtype; outputs and cache are always float32")
    parser.add_argument("--overlap", type=int, default=None,
                        help="Chunk overlap in residues (0 = non-overlapping ablation)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Process only the FIRST N rows (subset run: needs a separate --output-dir)")
    parser.add_argument("--nsc-ids", default=None,
                        help="Comma-separated nsc_id subset (subset run: needs a separate --output-dir)")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse a fingerprint-matching cache (default behaviour when one exists)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Discard an existing cache whose fingerprint differs")
    parser.add_argument("--export-csv", action="store_true",
                        help="Also write inspection CSVs (not the scientific artifacts)")
    parser.add_argument("--projection", action="store_true",
                        help="Apply a TRAINED projection checkpoint (fails if none is configured)")
    parser.add_argument("--cache-path", default=None,
                        help="Explicit cache path (fingerprint is always verified)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Opt in to trust_remote_code (recorded in the manifest and README)")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Never contact the Hugging Face Hub (offline machine)")
    parser.add_argument("--special-residue-policy", choices=["keep_if_supported", "replace_with_X", "error"],
                        default=None, help="Special residue handling policy")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    setup_logging(None)
    logger = get_logger("esmc_target.cli")

    try:
        config = load_config(args.config)
        full_run_dir = Path(config.output.directory)
        if args.dtype:
            config.runtime.dtype = args.dtype
        if args.overlap is not None:
            config.sequence.overlap = args.overlap
        if args.special_residue_policy:
            config.sequence.special_residue_policy = args.special_residue_policy
        if args.trust_remote_code:
            logger.warning("--trust-remote-code enabled: this is recorded in the manifest and README")
            config.model.trust_remote_code = True
        if args.local_files_only:
            config.model.local_files_only = True
        config.apply_overrides(output_dir=str(output_dir) if output_dir else None,
                               device=args.device, projection=args.projection,
                               export_csv=args.export_csv, cache_path=args.cache_path)

        nsc_ids = [n.strip() for n in args.nsc_ids.split(",") if n.strip()] if args.nsc_ids else None
        is_subset = bool(args.max_rows or nsc_ids)
        final_output = Path(config.output.directory)
        if is_subset:
            if not args.output_dir:
                raise IsolationError(
                    "--max-rows/--nsc-ids requires an explicit --output-dir that is not the full-run "
                    f"directory (configured: {full_run_dir}). Subset runs must never write into "
                    f"production outputs."
                )
            enforce_run_isolation(final_output, True, full_run_dir)
        if not is_subset and not args.output_dir:
            final_output = Path(config.output.directory)
        final_output.mkdir(parents=True, exist_ok=True)


        pipeline = PreprocessingPipeline(
            config=config,
            output_dir=final_output,
            input_path=args.input,
            max_rows=args.max_rows,
            nsc_ids=nsc_ids,
            resume=args.resume or not args.rebuild_cache,
            rebuild_cache=args.rebuild_cache,
            export_csv=args.export_csv,
            enable_projection=args.projection,
            cache_path=args.cache_path,
        )
        result = pipeline.run()
    except PipelineInterruptedError as exc:
        logger.error("INTERRUPTED: %s", exc)
        print(f"\nRUN INTERRUPTED\n{'=' * 70}\n{exc}\n{'=' * 70}\n", file=sys.stderr)
        return 130 if exc.shutdown_signal == "SIGINT" else 143
    except FatalError as exc:
        logger.error("FATAL: %s", exc)
        print(f"\nFATAL ERROR\n{'=' * 70}\n{exc}\n{'=' * 70}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        logger.error("Interrupted. The cache on disk is intact; re-run with --resume.")
        return 130

    print("\nArtifacts:")
    for label, path in (("drug embeddings", result.drug_embeddings_path),
                        ("protein cache", result.protein_cache_path),
                        ("audit package", result.target_embeddings_path),
                        ("manifest", result.output_dir / "manifest.json")):
        print(f"  {label:<18} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
