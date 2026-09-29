"""Artifact writers. Every file is written atomically (temp file + ``os.replace``).

Artifact roles (Section 14):

* ``protein_cache.pt``                    - computational cache
* ``drug_target_embeddings_1152.pt``      - the CancerCombo input (PRIMARY)
* ``target_embeddings.pt``                - audit / reproducibility package
* ``target_provenance.csv`` / ``drug_provenance.csv`` - human-readable audit trail
* ``failed_rows.csv``, ``manifest.json``, ``*_schema.json``
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .utils import atomic_path, atomic_write_json, atomic_write_text, get_logger, local_now_iso

LOGGER = get_logger("esmc_target.outputs")

PROVENANCE_BASE_COLUMNS = [
    "source_row_index", "nsc_id", "drug_name", "chembl_id", "target_id", "gene_symbol",
    "uniprot_id", "sequence_length", "actual_sequence_length", "sequence_length_mismatch",
    "sequence_hash", "embedding_cache_hit", "num_chunks", "special_residue_events",
    "embedding_l2_norm", "status", "error",
]
DRUG_PROVENANCE_COLUMNS = [
    "nsc_id", "drug_name", "num_target_proteins", "num_unique_target_proteins",
    "num_successful_targets", "num_failed_targets", "num_duplicate_targets_removed",
    "aggregation_method", "status", "error", "embedding_dim",
]


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return buffer.getvalue()


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    atomic_write_text(path, _csv_text(columns, rows))
    return path


def provenance_columns(extra_input_columns: Sequence[str]) -> List[str]:
    """Base provenance columns plus every other input metadata column, in order."""
    extras = [c for c in extra_input_columns
              if c not in PROVENANCE_BASE_COLUMNS and c != "sequence"]
    return PROVENANCE_BASE_COLUMNS + extras


def write_drug_embeddings(path: Path, drug_embeddings: Mapping[str, torch.Tensor]) -> Path:
    """Plain ``{nsc_id: Tensor[dim] float32}`` dict, written atomically."""
    payload = {k: v.detach().to(dtype=torch.float32, device="cpu").contiguous()
               for k, v in drug_embeddings.items()}
    with atomic_path(path) as tmp:
        torch.save(payload, tmp)
    return path


def write_target_embeddings(path: Path, row_embeddings: torch.Tensor,
                            row_embedding_row_indices: torch.Tensor,
                            row_valid_mask: torch.Tensor,
                            row_metadata: Sequence[Mapping[str, Any]],
                            drug_embeddings: Mapping[str, torch.Tensor],
                            drug_metadata: Mapping[str, Mapping[str, Any]],
                            config: Mapping[str, Any], model_name: str, model_revision: str,
                            embedding_dim: int, projection_enabled: bool,
                            aggregation_method: str) -> Path:
    """Full-provenance artifact.

    Row layout is COMPACT: ``row_embeddings`` holds only successful rows, with
    ``row_embedding_row_indices[k]`` giving the original input row index. No NaN,
    zero, random or placeholder tensor is ever created for a failed row, and the
    successful-only tensor is never indexed directly against all-rows metadata.
    """
    n_input = len(row_metadata)
    payload = {
        "row_embeddings": row_embeddings.to(dtype=torch.float32, device="cpu").contiguous(),
        "row_embedding_row_indices": row_embedding_row_indices.to(dtype=torch.long, device="cpu"),
        "row_valid_mask": row_valid_mask.to(dtype=torch.bool, device="cpu"),
        "row_metadata": [dict(m) for m in row_metadata],
        "drug_embeddings": {k: v.detach().to(dtype=torch.float32, device="cpu").contiguous()
                            for k, v in drug_embeddings.items()},
        "drug_metadata": {k: dict(v) for k, v in drug_metadata.items()},
        "config": dict(config),
        "model_name": model_name,
        "model_revision": model_revision,
        "embedding_dim": int(embedding_dim),
        "projection_enabled": bool(projection_enabled),
        "aggregation_method": aggregation_method,
        "row_layout": {
            "description": "row_embeddings is compact (successful rows only); "
                           "row_embedding_row_indices[k] is the input row index of row_embeddings[k]; "
                           "row_valid_mask has one entry per input row.",
            "n_input_rows": int(n_input),
            "n_successful_rows": int(row_embeddings.shape[0]),
        },
        "written_at": local_now_iso(),
    }
    with atomic_path(path) as tmp:
        torch.save(payload, tmp)
    return path


def build_schema(model_id: str, model_revision: str, hidden_size: int,
                 projection_applied: bool, layernorm_applied: bool) -> Dict[str, Any]:
    """Machine-readable schema written next to the drug embedding .pt file.

    All values are generated at runtime; nothing here is a hardcoded template.
    Downstream code must never assume ``key == drug_name``, ``dim == 256`` or
    that LayerNorm was already applied.
    """
    return {
        "embedding_dim": int(hidden_size),
        "key_type": "nsc_id",
        "dtype": "float32",
        "aggregation": "mean",
        "model": "ESMC-600M",
        "model_id": model_id,
        "model_revision": model_revision,
        "sequence_identity": "SHA256(cleaned_sequence)",
        "layernorm_applied": bool(layernorm_applied),
        "projection_applied": bool(projection_applied),
    }


def write_schema(path: Path, schema: Mapping[str, Any]) -> Path:
    atomic_write_json(path, dict(schema))
    return path


def embedding_filename(hidden_size: int, projection_enabled: bool) -> str:
    return f"drug_target_embeddings_{256 if projection_enabled else hidden_size}.pt"


def write_embedding_csv(path: Path, rows: Sequence[Mapping[str, Any]],
                        embeddings: Sequence[Sequence[float]], embedding_dim: int,
                        stage: str) -> Path:
    """Inspection-only CSV: provenance + ``emb_1..emb_D`` + ``embedding_stage``."""
    columns = list(rows[0].keys()) if rows else []
    columns = columns + ["embedding_dim", "embedding_stage"] + [f"emb_{i + 1}" for i in range(embedding_dim)]
    materialised: List[Dict[str, Any]] = []
    for row, vector in zip(rows, embeddings):
        entry = dict(row)
        entry["embedding_dim"] = int(embedding_dim)
        entry["embedding_stage"] = stage
        for i in range(embedding_dim):
            entry[f"emb_{i + 1}"] = vector[i] if i < len(vector) else None
        materialised.append(entry)
    return write_csv(path, columns, materialised)


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    atomic_write_json(path, dict(manifest))
    LOGGER.info("Wrote manifest: %s", path)
    return path


def write_diagnostics(path: Path, diagnostics: Mapping[str, Any]) -> Path:
    atomic_write_json(path, dict(diagnostics))
    return path


def write_failed_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """``failed_rows.csv`` (may legitimately be empty apart from the header)."""
    columns = PROVENANCE_BASE_COLUMNS
    return write_csv(path, columns, rows)


def summarise_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        summary[status] = summary.get(status, 0) + 1
    return summary
