"""Drug-level aggregation: MEAN over a drug's unique successful target proteins.

Pooling point **B** (protein -> drug target embedding) lives here. It is
parameter-free, deterministic, permutation-invariant and label-free.

Two deliberate design decisions:

1. **Deduplicate by ``sequence_hash`` within each drug** before averaging, so a
   sequence annotated several times cannot be counted twice. The count removed
   is recorded.
2. **No attention aggregator is implemented.** This stage is offline, label-free
   feature extraction with no loss and no training loop, so learnable weights
   would be untrained noise presented as a learned weighting. Target-level
   attention/gating is a supervised ablation inside CancerCombo.

Equal weighting is a *computational representation assumption*, not a biological
claim that all annotated targets matter equally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .errors import FatalError
from .utils import get_logger

LOGGER = get_logger("esmc_target.aggregation")

VALID_METHODS = ("mean",)


@dataclass
class DrugAggregation:
    nsc_id: str
    embedding: torch.Tensor
    drug_name: Optional[str] = None
    num_target_proteins: int = 0
    num_unique_target_proteins: int = 0
    num_successful_targets: int = 0
    num_failed_targets: int = 0
    num_duplicate_targets_removed: int = 0
    unique_sequence_hashes: List[str] = field(default_factory=list)

    def metadata(self, embedding_dim: int) -> Dict[str, Any]:
        return {
            "nsc_id": self.nsc_id,
            "drug_name": self.drug_name,
            "num_target_proteins": self.num_target_proteins,
            "num_unique_target_proteins": self.num_unique_target_proteins,
            "num_successful_targets": self.num_successful_targets,
            "num_failed_targets": self.num_failed_targets,
            "num_duplicate_targets_removed": self.num_duplicate_targets_removed,
            "aggregation_method": "mean",
            "status": "success",
            "error": None,
            "embedding_dim": int(embedding_dim),
        }


@dataclass
class AggregationResult:
    drug_embeddings: Dict[str, torch.Tensor]
    drug_metadata: Dict[str, Dict[str, Any]]
    drugs_without_successful_targets: List[str]
    within_drug_duplicates_removed: int
    method: str = "mean"

    def summary(self) -> Dict[str, Any]:
        return {
            "num_drugs_with_embeddings": len(self.drug_embeddings),
            "drugs_without_successful_targets": self.drugs_without_successful_targets,
            "within_drug_duplicates_removed": self.within_drug_duplicates_removed,
            "aggregation_method": self.method,
        }


def aggregate_drug_embeddings(rows: Sequence[Any], embeddings_by_hash: Dict[str, torch.Tensor],
                              method: str = "mean",
                              embedding_dim: Optional[int] = None) -> AggregationResult:
    """Group successful rows by ``nsc_id``, dedupe by sequence hash, then mean.

    ``rows`` are :class:`esmc_target.data.RowRecord` objects. A drug with zero
    successful targets is **omitted** from the output dict (so CancerCombo can
    derive its modality mask from key absence); it never receives a zero/NaN
    vector.
    """
    if method not in VALID_METHODS:
        raise FatalError(
            f"aggregation.method must be one of {list(VALID_METHODS)}, got {method!r}. "
            f"No silent fallback and no attention aggregator in this repository."
        )

    grouped: Dict[str, List[Any]] = {}
    for row in rows:
        grouped.setdefault(row.nsc_id, []).append(row)

    drug_embeddings: Dict[str, torch.Tensor] = {}
    drug_metadata: Dict[str, Dict[str, Any]] = {}
    without: List[str] = []
    duplicates_removed = 0

    for nsc_id in sorted(grouped):
        drug_rows = grouped[nsc_id]
        drug_name = next((r.drug_name for r in drug_rows if r.drug_name), None)
        failed = [r for r in drug_rows if not r.ok]
        successful = [r for r in drug_rows if r.ok and r.sequence_hash in embeddings_by_hash]

        unique_hashes: List[str] = []
        vectors: List[torch.Tensor] = []
        seen: set = set()
        for row in successful:
            if row.sequence_hash in seen:
                continue
            seen.add(row.sequence_hash)
            unique_hashes.append(row.sequence_hash)
            vectors.append(embeddings_by_hash[row.sequence_hash].to(dtype=torch.float32))
        duplicates_removed += len(successful) - len(unique_hashes)

        dim = embedding_dim or (int(vectors[0].shape[0]) if vectors else 0)
        meta = {
            "nsc_id": nsc_id,
            "drug_name": drug_name,
            "num_target_proteins": len(drug_rows),
            "num_unique_target_proteins": len(seen),
            "num_successful_targets": len(successful),
            "num_failed_targets": len(failed),
            "num_duplicate_targets_removed": len(successful) - len(unique_hashes),
            "aggregation_method": "mean",
            "status": "success" if vectors else "failed",
            "error": None if vectors else (
                "no successful target protein embeddings for this drug; omitted from the "
                "embedding dict so the downstream modality mask can be derived from key absence"
            ),
            "embedding_dim": int(dim) if vectors else 0,
        }

        if not vectors:
            without.append(nsc_id)
            drug_metadata[nsc_id] = meta
            LOGGER.warning("Drug %s has no successful target proteins: omitted from the dict", nsc_id)
            continue

        stacked = torch.stack(vectors, dim=0)
        mean = stacked.mean(dim=0).to(dtype=torch.float32, device="cpu").contiguous()
        if not torch.isfinite(mean).all():
            raise FatalError(f"aggregated embedding for drug {nsc_id} contains NaN/Inf")
        if float(mean.norm().item()) <= 0.0:
            raise FatalError(f"aggregated embedding for drug {nsc_id} has zero norm")
        drug_embeddings[nsc_id] = mean
        drug_metadata[nsc_id] = meta

    return AggregationResult(
        drug_embeddings=drug_embeddings,
        drug_metadata=drug_metadata,
        drugs_without_successful_targets=sorted(without),
        within_drug_duplicates_removed=duplicates_removed,
        method=method,
    )
