"""Input reading, canonical column mapping and per-row sequence preparation.

Rules:

* Column matching is case-insensitive; internally everything is lowercase.
  Every input column is preserved as metadata (no fixed whitelist, no invented
  provenance fields).
* Required columns are ``nsc_id`` and ``sequence``; a missing one raises a clear,
  named error - never a raw KeyError.
* Grouping is by ``nsc_id``; ``drug_name`` is metadata. Protein identity is
  sequence-based only (``target_id`` is not a safe key: the current file has 244
  unique target_id but 245 unique sequences).
* The ``sequence_length`` column is metadata. A mismatch is a *warning*, never a
  failed row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .errors import FatalError, RowError
from .identifiers import canonicalize_nsc_id
from .sequence import (
    CleanResult,
    clean_sequence,
    sequence_hash,
    special_residue_occurrences,
    validate_declared_length,
)
from .utils import sha256_file

REQUIRED_COLUMNS = ("nsc_id", "sequence")
#: Columns that get dedicated provenance handling (all other columns still flow
#: through as generic metadata).
KNOWN_COLUMNS = (
    "nsc_id", "drug_name", "chembl_id", "target_id", "gene_symbol", "uniprot_id",
    "sequence", "sequence_length", "target_source", "evidence",
)


@dataclass
class InputFile:
    path: Path
    sha256: str
    size_bytes: int
    frame: pd.DataFrame

    @property
    def n_rows(self) -> int:
        return int(self.frame.shape[0])


@dataclass
class RowRecord:
    """One input row after canonicalization, cleaning and hashing."""

    source_row_index: int
    nsc_id: str
    sequence_raw: Optional[str]
    sequence: Optional[str] = None
    sequence_hash: Optional[str] = None
    declared_length: Optional[int] = None
    actual_length: Optional[int] = None
    length_mismatch: bool = False
    status: str = "pending"
    error: Optional[str] = None
    error_kind: Optional[str] = None
    warning: Optional[str] = None
    cache_hit: bool = False
    num_chunks: int = 0
    embedding_l2_norm: Optional[float] = None
    special_residue_events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    uniprot_id: Optional[str] = None
    target_id: Optional[str] = None
    gene_symbol: Optional[str] = None
    drug_name: Optional[str] = None
    sequence_original: Optional[str] = None
    special_residue_symbols: List[str] = field(default_factory=list)
    special_residue_symbols_after_cleaning: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "success"


def canonicalize_columns(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Lowercase column names; return ``(frame, original->canonical map, collisions)``."""
    mapping: Dict[str, str] = {}
    collisions: List[str] = []
    new_names: List[str] = []
    seen: Dict[str, int] = {}
    for original in frame.columns:
        canonical = str(original).strip().lower()
        if canonical in seen:
            collisions.append(canonical)
            canonical = f"{canonical}__{seen[canonical]}"
        seen[canonical] = seen.get(canonical, 0) + 1
        mapping[str(original)] = canonical
        new_names.append(canonical)
    out = frame.copy()
    out.columns = new_names
    return out, mapping, collisions


def read_input(path: str | Path) -> InputFile:
    """Read the CSV and hash the **raw file bytes** (never a dataframe)."""
    path = Path(path)
    if not path.exists():
        raise FatalError(f"Input file not found: {path}")
    if path.stat().st_size == 0:
        raise FatalError(f"Input file is empty: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    except Exception as exc:
        raise FatalError(f"Could not parse {path} as CSV: {exc}")
    frame, _mapping, collisions = canonicalize_columns(frame)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise FatalError(
            f"Input file {path.name} is missing required column(s): {missing}. "
            f"Found columns: {list(frame.columns)}. Required: {list(REQUIRED_COLUMNS)}."
        )
    if collisions:
        raise FatalError(
            f"Input file {path.name} has columns that collide after case-insensitive "
            f"normalization: {collisions}. Rename them to be unique."
        )
    return InputFile(path=path, sha256=sha256_file(path), size_bytes=path.stat().st_size, frame=frame)


def filter_rows(frame: pd.DataFrame, max_rows: Optional[int] = None,
                nsc_ids: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply ``--max-rows`` (FIRST N rows only) and ``--nsc-ids``.
    
    Preserves the original dataframe index (source_row_index) for provenance auditability.
    """
    out = frame
    applied: Dict[str, Any] = {"max_rows": None, "nsc_ids": None}
    if nsc_ids:
        canonical_wanted = []
        for n in nsc_ids:
            s = str(n).strip()
            if s:
                try:
                    canonical_wanted.append(canonicalize_nsc_id(s))
                except ValueError:
                    canonical_wanted.append(s.upper())
        if not canonical_wanted:
            raise FatalError("--nsc-ids was provided but contains no usable id")

        def match_nsc(val: Any) -> bool:
            try:
                return canonicalize_nsc_id(val) in canonical_wanted
            except ValueError:
                return False

        mask = out["nsc_id"].apply(match_nsc)
        out = out[mask]
        applied["nsc_ids"] = sorted(canonical_wanted)
        if out.empty:
            raise FatalError(f"--nsc-ids {sorted(canonical_wanted)} matched no rows in the input file")
    if max_rows is not None:
        if max_rows < 1:
            raise FatalError(f"--max-rows must be >= 1, got {max_rows}")
        out = out.head(max_rows)
        applied["max_rows"] = int(max_rows)
    # Note: Do NOT reset_index(drop=True) so original index is preserved
    return out, applied


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    return text if text else None


def prepare_rows(frame: pd.DataFrame, special_residue_policy: str,
                 tokenizer_support: Optional[Dict[str, bool]] = None) -> List[RowRecord]:
    """Clean + hash every row. Invalid sequences become failed rows, never drops."""
    records: List[RowRecord] = []
    for index, row in frame.iterrows():
        raw_nsc = row.get("nsc_id")
        nsc_error = None
        canonical_nsc = ""
        try:
            if raw_nsc is not None and str(raw_nsc).strip():
                canonical_nsc = canonicalize_nsc_id(raw_nsc)
            else:
                nsc_error = "nsc_id is missing or empty; a row cannot be grouped to a drug"
        except ValueError as exc:
            nsc_error = str(exc)

        metadata = {k: _as_text(v) for k, v in row.items() if k != "sequence"}
        record = RowRecord(
            source_row_index=int(index),
            nsc_id=canonical_nsc,
            sequence_raw=_as_text(row.get("sequence")),
            metadata=metadata,
            uniprot_id=_as_text(row.get("uniprot_id")),
            target_id=_as_text(row.get("target_id")),
            gene_symbol=_as_text(row.get("gene_symbol")),
            drug_name=_as_text(row.get("drug_name")),
        )
        if nsc_error or not record.nsc_id:
            record.status = "failed"
            record.error = nsc_error or "nsc_id is missing or empty; a row cannot be grouped to a drug"
            record.error_kind = "invalid_nsc" if nsc_error else "missing_key"
            records.append(record)
            continue
        try:
            clean: CleanResult = clean_sequence(record.sequence_raw, policy=special_residue_policy,
                                                 tokenizer_support=tokenizer_support)
        except RowError as exc:
            record.status = "failed"
            record.error = str(exc)
            record.error_kind = getattr(exc, "kind", "row_error")
            records.append(record)
            continue
        except Exception as exc:  # defensive: still a row-level failure
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            record.error_kind = "cleaning_error"
            records.append(record)
            continue

        record.sequence = clean.cleaned
        record.sequence_original = clean.original
        record.sequence_hash = sequence_hash(clean.cleaned)
        record.actual_length = clean.length
        record.special_residue_events = clean.events_payload()
        # Symbols are reported from the ORIGINAL sequence so that a residue that
        # was transparently replaced with X is still visible as U in provenance.
        record.special_residue_symbols = sorted({c for _p, c in special_residue_occurrences(clean.original)})
        record.special_residue_symbols_after_cleaning = sorted(
            {c for _p, c in special_residue_occurrences(clean.cleaned)})
        declared, mismatch, warning = validate_declared_length(row.get("sequence_length"), clean.length)
        record.declared_length = declared
        record.length_mismatch = mismatch
        record.warning = warning
        record.status = "pending"
        records.append(record)
    return records


def dataset_statistics(records: Sequence[RowRecord], residue_capacity: Optional[int] = None) -> Dict[str, Any]:
    """Dataset statistics, all computed dynamically (never hardcoded)."""
    cleaned = [r for r in records if r.sequence]
    lengths = [r.actual_length or 0 for r in cleaned]
    hashes = {r.sequence_hash for r in cleaned}
    long_rows = [r for r in cleaned if (r.actual_length or 0) > 2048]
    stats: Dict[str, Any] = {
        "input_rows": len(records),
        "rows_with_valid_sequence": len(cleaned),
        "unique_nsc_id": len({r.nsc_id for r in records if r.nsc_id}),
        "unique_drug_name": len({r.drug_name for r in records if r.drug_name}),
        "unique_target_id": len({r.target_id for r in records if r.target_id}),
        "unique_uniprot_id": len({r.uniprot_id for r in records if r.uniprot_id}),
        "unique_sequences": len(hashes),
        "sequence_length_min": min(lengths) if lengths else None,
        "sequence_length_max": max(lengths) if lengths else None,
        "sequence_length_mean": (sum(lengths) / len(lengths)) if lengths else None,
        "rows_with_length_mismatch": sum(1 for r in records if r.length_mismatch),
        "rows_with_missing_or_invalid_declared_length": sum(
            1 for r in records if r.sequence and r.declared_length is None),
        "rows_missing_sequence": sum(1 for r in records if not r.sequence),
        "duplicate_sequence_rows": len(cleaned) - len(hashes),
        "rows_over_2048_residues": len(long_rows),
        "unique_sequences_over_2048_residues": len({r.sequence_hash for r in long_rows}),
        "sequences_with_nonstandard_residues": [
            {"nsc_id": r.nsc_id, "gene_symbol": r.gene_symbol, "uniprot_id": r.uniprot_id,
             "sequence_length": r.actual_length,
             "symbols_in_input": r.special_residue_symbols,
             "symbols_after_cleaning": r.special_residue_symbols_after_cleaning,
             "num_transformations": len(r.special_residue_events),
             "sequence_hash": r.sequence_hash}
            for r in cleaned if r.special_residue_symbols
        ],
    }
    if residue_capacity:
        need = [r for r in cleaned if (r.actual_length or 0) > residue_capacity]
        stats["residue_capacity_used_for_long_check"] = int(residue_capacity)
        stats["unique_sequences_requiring_chunking"] = len({r.sequence_hash for r in need})
        stats["rows_requiring_chunking"] = len(need)
        stats["long_sequences_detail"] = [
            {"nsc_id": r.nsc_id, "gene_symbol": r.gene_symbol, "uniprot_id": r.uniprot_id,
             "sequence_length": r.actual_length, "sequence_hash": r.sequence_hash}
            for r in need
        ]
    return stats
