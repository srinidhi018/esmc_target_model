"""OPTIONAL downstream projection: ``Linear(1152 -> 256)`` + ``LayerNorm(256)``.

**Disabled by default.** This repository does not train projections. A
projection checkpoint must come from a *trained* CancerCombo run; random weights
are never created and never labelled as trained.

Ordering matters: the linear map commutes with a mean, so averaging in 1152-D
then projecting equals projecting then averaging. ``LayerNorm`` does **not**
commute with averaging, so it must always be applied to the **aggregate**, never
per protein. A projection frozen offline and untrained could not be corrected
later, which is why the default pipeline keeps the full 1152-D vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

from .errors import ProjectionError


class TargetProjection(nn.Module):
    """``Linear(input_dim -> output_dim)`` followed by ``LayerNorm(output_dim)``.

    ``input_dim`` is always the resolved ``model.config.hidden_size`` (asserted
    == 1152 for ESMC-600M); it is never a hand-configured number.
    """

    def __init__(self, input_dim: int = 1152, output_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project an already-aggregated 1152-D vector (or a batch of them)."""
        if x.shape[-1] != self.input_dim:
            raise ProjectionError(
                f"TargetProjection expects input width {self.input_dim}, got {tuple(x.shape)}")
        return self.layer_norm(self.linear(x))

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"


def load_projection(checkpoint_path: str | Path, input_dim: int = 1152,
                    output_dim: int = 256) -> TargetProjection:
    """Load a genuinely trained projection checkpoint. Fails clearly otherwise."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise ProjectionError(
            f"projection.checkpoint not found: {path}. This repository does not train "
            f"projections; export one from a trained CancerCombo run or keep "
            f"projection.enabled: false. Random weights are never used as a substitute."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ProjectionError(f"projection checkpoint {path} could not be read: {exc}")

    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise ProjectionError(
            f"projection checkpoint {path} has no 'state_dict' key; refusing to guess a layout."
        )
    meta: Dict[str, Any] = dict(payload.get("metadata") or {})
    if not meta.get("trained", False):
        raise ProjectionError(
            f"projection checkpoint {path} is not marked as trained (metadata.trained != True). "
            f"This pipeline will not present untrained weights as a learned projection."
        )
    checkpoint_input = int(meta.get("input_dim", input_dim))
    checkpoint_output = int(meta.get("output_dim", output_dim))
    if checkpoint_input != input_dim or checkpoint_output != output_dim:
        raise ProjectionError(
            f"projection checkpoint {path} was trained for {checkpoint_input}->{checkpoint_output} "
            f"but the current pipeline requires {input_dim}->{output_dim}."
        )

    projection = TargetProjection(input_dim=input_dim, output_dim=output_dim)
    state = payload["state_dict"]
    try:
        projection.load_state_dict(state)
    except Exception as exc:
        raise ProjectionError(
            f"projection checkpoint {path} does not match TargetProjection({input_dim}, "
            f"{output_dim}): {exc}")
    projection.eval()
    for param in projection.parameters():
        param.requires_grad_(False)
    return projection


def project_embeddings(drug_embeddings: Mapping[str, torch.Tensor],
                       projection: TargetProjection) -> Dict[str, torch.Tensor]:
    """Apply Linear+LayerNorm to each **aggregated** drug vector (never per protein)."""
    out: Dict[str, torch.Tensor] = {}
    for key, vector in drug_embeddings.items():
        with torch.no_grad():
            projected = projection(vector.to(dtype=torch.float32).reshape(1, -1))
        out[key] = projected.reshape(-1).to(dtype=torch.float32, device="cpu").contiguous()
    return out
