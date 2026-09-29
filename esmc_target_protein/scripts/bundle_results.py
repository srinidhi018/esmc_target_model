#!/usr/bin/env python
"""Collect run logs, manifests, provenance CSVs and diagnostics into one zip.

Large ``.pt`` artifacts are included only with ``--include-embeddings``.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from esmc_target.utils import local_now_iso, setup_logging  # noqa: E402

ALWAYS = ["manifest.json", "run.log", "target_provenance.csv", "drug_provenance.csv",
          "failed_rows.csv", "embedding_diagnostics.json", "drug_target_embeddings_schema.json",
          "target_embeddings.csv", "drug_target_embeddings.csv"]
EMBEDDINGS = ["protein_cache.pt", "target_embeddings.pt", "drug_target_embeddings_1152.pt",
              "drug_target_embeddings_256.pt"]


def collect(output_dirs, out_path: Path, include_embeddings: bool) -> int:
    wanted = set(ALWAYS) | (set(EMBEDDINGS) if include_embeddings else set())
    written = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BUNDLE_INFO.txt",
                    f"created_at: {local_now_iso()}\n"
                    f"include_embeddings: {include_embeddings}\n"
                    f"output_dirs: {[str(d) for d in output_dirs]}\n")
        for directory in output_dirs:
            directory = Path(directory)
            if not directory.exists():
                continue
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.name not in wanted:
                    continue
                zf.write(path, arcname=f"{directory.name}/{path.name}")
                written += 1
        log_dir = REPO_ROOT / "outputs" / "logs"
        if log_dir.exists():
            for path in sorted(log_dir.iterdir()):
                if path.is_file():
                    zf.write(path, arcname=f"logs/{path.name}")
                    written += 1
        preflight = REPO_ROOT / "outputs" / "preflight_report.json"
        if preflight.exists():
            zf.write(preflight, arcname="preflight_report.json")
            written += 1
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bundle run artifacts for review.")
    parser.add_argument("--output-dir", action="append", default=None,
                        help="Run directory to include (repeatable; default outputs/full + edge + smoke)")
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "results_bundle.zip"))
    parser.add_argument("--include-embeddings", action="store_true",
                        help="Include the large .pt artifacts (off by default)")
    args = parser.parse_args(argv)
    logger = setup_logging(None)
    dirs = args.output_dir or [str(REPO_ROOT / "outputs" / name)
                               for name in ("full", "edge_cases", "smoke_test")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = collect(dirs, out, args.include_embeddings)
    logger.info("Wrote %s (%d files)", out, count)
    print(f"Wrote {out} ({count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
