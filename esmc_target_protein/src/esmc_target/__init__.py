"""Offline ESMC-600M target-protein preprocessing and embedding pipeline.

    targetprotein.csv -> canonicalize -> clean -> SHA256 -> frozen ESMC-600M
    -> residue-aligned hidden states -> residue mean (overlapping chunks)
    -> protein embedding [1152] -> cache -> mean over a drug's unique targets
    -> drug target embedding [1152]

No chemistry encoders, no pathways, no projection to 256-D by default, no
training of any kind, and no online lookups.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "aggregation",
    "cache",
    "config",
    "data",
    "diagnostics",
    "errors",
    "esmc_encoder",
    "outputs",
    "pipeline",
    "pooling",
    "projection",
    "sequence",
    "utils",
]
