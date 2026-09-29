#!/usr/bin/env python
"""Read-only dataset inspection. Never modifies the dataset.

Reports rows, columns exactly as found, unique ids, sequence length statistics,
length mismatches, missing/duplicate sequences, long sequences (over 2048
residues, and over the derived residue capacity when a model is available), rows
with non-standard residues, and within-drug duplicate sequences.

Everything is computed dynamically; the expected statistics of Section 12 are
acceptance-test values, not pipeline invariants.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from esmc_target.data import (  # noqa: E402
    canonicalize_columns,
    dataset_statistics,
    filter_rows,
    prepare_rows,
    read_input,
)
from esmc_target.sequence import SPECIAL_RESIDUES, SPECIAL_RESIDUE_SET  # noqa: E402
from esmc_target.utils import setup_logging  # noqa: E402


def inspect(path: str, max_rows: int | None, nsc_ids: str | None) -> dict:
    source = read_input(path)
    frame, applied = filter_rows(source.frame, max_rows, nsc_ids)
    canonical_frame, mapping, _collisions = canonicalize_columns(frame)
    records = prepare_rows(canonical_frame, "keep_if_supported",
                           tokenizer_support={s: True for s in SPECIAL_RESIDUES})
    cleaned = [r for r in records if r.sequence]
    counts = Counter(r.sequence_hash for r in cleaned)

    within_drug: Counter = Counter()
    for record in cleaned:
        within_drug[(record.nsc_id, record.sequence_hash)] += 1

    over_2048 = [r for r in cleaned if (r.actual_length or 0) > 2048]
    specials = [r for r in cleaned if any(c in SPECIAL_RESIDUE_SET for c in r.sequence)]

    report = {
        "input_file": str(source.path),
        "input_sha256": source.sha256,
        "input_size_bytes": source.size_bytes,
        "columns_as_found": list(source.frame.columns),
        "column_name_mapping": mapping,
        "subset_filter_applied": applied,
        "statistics": dataset_statistics(records),
        "duplicate_sequence_hashes": {h: c for h, c in counts.items() if c > 1},
        "within_drug_duplicate_sequences": [
            {"nsc_id": nsc, "sequence_hash": h, "occurrences": c}
            for (nsc, h), c in within_drug.items() if c > 1
        ],
        "rows_over_2048_residues": [
            {"source_row_index": r.source_row_index, "nsc_id": r.nsc_id,
             "gene_symbol": r.gene_symbol, "uniprot_id": r.uniprot_id,
             "sequence_length": r.actual_length, "sequence_hash": r.sequence_hash}
            for r in over_2048
        ],
        "rows_with_nonstandard_residues": [
            {"source_row_index": r.source_row_index, "nsc_id": r.nsc_id,
             "gene_symbol": r.gene_symbol, "uniprot_id": r.uniprot_id,
             "sequence_length": r.actual_length,
             "symbols": sorted({c for c in r.sequence if c in SPECIAL_RESIDUE_SET}),
             "positions": {s: [i for i, c in enumerate(r.sequence) if c == s]
                           for s in sorted({c for c in r.sequence if c in SPECIAL_RESIDUE_SET})}}
            for r in specials
        ],
        "rows_with_length_mismatch": [
            {"source_row_index": r.source_row_index, "nsc_id": r.nsc_id,
             "declared": r.declared_length, "actual": r.actual_length}
            for r in records if r.length_mismatch
        ],
        "rows_failed_cleaning": [
            {"source_row_index": r.source_row_index, "nsc_id": r.nsc_id,
             "error": r.error, "error_kind": r.error_kind}
            for r in records if r.status == "failed"
        ],
        "residue_capacity_note": (
            "Capacity-dependent counts (unique sequences requiring chunking) need the resolved "
            "model; run scripts/check_environment.py or the full pipeline to obtain "
            "residue_capacity = model.config.max_position_embeddings - special tokens."
        ),
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect targetprotein.csv (read-only).")
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "targetprotein.csv"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--nsc-ids", default=None)
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Write the report as JSON to this path")
    args = parser.parse_args(argv)
    setup_logging(None)
    nsc_ids = [n.strip() for n in args.nsc_ids.split(",") if n.strip()] if args.nsc_ids else None
    report = inspect(args.input, args.max_rows, nsc_ids)
    if args.json_out:
        from esmc_target.utils import atomic_write_json
        atomic_write_json(args.json_out, report)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
