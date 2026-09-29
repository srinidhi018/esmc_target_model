"""Residue-aligned mean pooling, chunk accumulation and embedding diagnostics.

Pooling point **A** (residue -> protein) lives here and is parameter-free:
mean over the real amino-acid residues only, never over CLS/EOS/PAD. The
position-averaged accumulator below implements pooling point **A** for long
proteins: per-residue sums are accumulated across overlapping chunks and
divided by the per-position coverage count, so every residue contributes
exactly once regardless of how many chunks saw it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .errors import EmbeddingQualityError


@dataclass
class PoolingDiagnostics:
    embedding_mean: float
    embedding_std: float
    embedding_l2_norm: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "embedding_mean": self.embedding_mean,
            "embedding_std": self.embedding_std,
            "embedding_l2_norm": self.embedding_l2_norm,
        }


def residue_mean_pool(residue_hidden_states: torch.Tensor) -> torch.Tensor:
    """Mean over residue positions only.

    ``residue_hidden_states`` must already be residue-aligned, i.e. special
    tokens removed by the encoder. This function is deliberately blind to
    whether the source field was ``last_hidden_state``, ``hidden_states[-1]``
    or anything else.
    """
    if residue_hidden_states.dim() != 2:
        raise ValueError(
            f"expected residue hidden states of shape [num_residues, hidden_size], got "
            f"{tuple(residue_hidden_states.shape)}"
        )
    if residue_hidden_states.shape[0] == 0:
        raise EmbeddingQualityError("cannot pool an empty residue tensor (zero residues)")
    return residue_hidden_states.mean(dim=0)


class ResidueAccumulator:
    """Per-residue sum/count accumulators across chunks.

    Guarantees the assertion of Section 8: every residue position has
    ``count >= 1`` and the number of pooled residues equals the sequence length.
    """

    def __init__(self, length: int, hidden_size: int, device: Optional[torch.device] = None) -> None:
        self.length = int(length)
        self.hidden_size = int(hidden_size)
        self.device = device or torch.device("cpu")
        self.sum: torch.Tensor = torch.zeros((self.length, self.hidden_size), dtype=torch.float32,
                                             device=self.device)
        self.count: torch.Tensor = torch.zeros((self.length,), dtype=torch.float32, device=self.device)
        self.chunk_lengths: List[int] = []

    def add(self, start: int, end: int, residue_states: torch.Tensor) -> None:
        if residue_states.shape[0] != end - start:
            raise ValueError(
                f"chunk [{start}, {end}) expects {end - start} residue vectors, got "
                f"{residue_states.shape[0]}"
            )
        self.sum[start:end] += residue_states.to(dtype=torch.float32, device=self.device)
        self.count[start:end] += 1.0
        self.chunk_lengths.append(int(end - start))

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_lengths)

    @property
    def total_chunk_residues(self) -> int:
        return int(sum(self.chunk_lengths))

    def assert_full_coverage(self) -> Dict[str, int]:
        uncovered = int((self.count < 1).sum().item())
        if uncovered > 0:
            raise EmbeddingQualityError(
                f"{uncovered} residue position(s) were never encoded: no chunk covered them. "
                f"Nothing is silently dropped."
            )
        if int(self.count.shape[0]) != self.length:
            raise EmbeddingQualityError(
                f"accumulator covers {int(self.count.shape[0])} positions, expected {self.length}"
            )
        return {
            "num_residues_pooled": int(self.length),
            "num_chunks": self.num_chunks,
            "total_chunk_residues": self.total_chunk_residues,
            "min_coverage_count": int(self.count.min().item()),
            "max_coverage_count": int(self.count.max().item()),
        }

    def per_residue_embeddings(self) -> torch.Tensor:
        self.assert_full_coverage()
        return self.sum / self.count.unsqueeze(1)

    def pooled(self) -> torch.Tensor:
        """Protein embedding: mean over all L per-residue embeddings."""
        return self.per_residue_embeddings().mean(dim=0)


def pool_chunks(chunk_residues: Sequence[Tuple[int, int, torch.Tensor]], length: int,
                hidden_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Pool a list of ``(start, end, residue_states)`` chunks into one vector."""
    acc = ResidueAccumulator(length=length, hidden_size=hidden_size, device=device)
    for start, end, states in chunk_residues:
        acc.add(start, end, states)
    coverage = acc.assert_full_coverage()
    return acc.pooled(), coverage


def compute_diagnostics(embedding: torch.Tensor) -> PoolingDiagnostics:
    """Section 9 numerical sanity diagnostics for one protein embedding."""
    flat = embedding.detach().to(dtype=torch.float32).flatten()
    return PoolingDiagnostics(
        embedding_mean=float(flat.mean().item()),
        embedding_std=float(flat.std(unbiased=False).item()),
        embedding_l2_norm=float(flat.norm().item()),
    )


def validate_embedding(embedding: torch.Tensor, hidden_size: int,
                       sequence_hash: Optional[str] = None) -> None:
    """Reject NaN/Inf/zero/wrong-dimension embeddings. Raising -> row failure."""
    prefix = f"sequence_hash={sequence_hash}: " if sequence_hash else ""
    if not isinstance(embedding, torch.Tensor):
        raise EmbeddingQualityError(f"{prefix}embedding is a {type(embedding).__name__}, not a Tensor")
    if embedding.dim() != 1:
        raise EmbeddingQualityError(
            f"{prefix}expected a 1-D protein embedding, got shape {tuple(embedding.shape)}"
        )
    if int(embedding.shape[0]) != int(hidden_size):
        raise EmbeddingQualityError(
            f"{prefix}embedding dim {int(embedding.shape[0])} != model.config.hidden_size "
            f"{int(hidden_size)}"
        )
    if not torch.isfinite(embedding).all():
        raise EmbeddingQualityError(f"{prefix}embedding contains NaN or Inf")
    norm = float(embedding.detach().to(torch.float32).norm().item())
    if norm <= 0.0:
        raise EmbeddingQualityError(f"{prefix}embedding L2 norm is {norm}; refusing a zero vector")
