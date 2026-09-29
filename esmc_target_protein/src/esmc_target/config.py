"""Typed configuration: YAML load, validation, aliasing, hashing, CLI overrides.

Every value the pipeline uses is read from here; nothing scientific is
hardcoded in the modules. The one non-configurable number is the ESMC-600M
hidden size, which is *asserted* after model verification and from which all
tensor shapes, cache schemas, CSV column counts and projection ``input_dim``
are derived.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .errors import ConfigError
from .utils import sha256_json

#: Acceptance invariant of the required ESMC-600M backbone. Not a free
#: parameter: it is asserted once, after the checkpoint has been verified.
REQUIRED_HIDDEN_SIZE = 1152
#: Dimensionality of the optional downstream projection (learned in CancerCombo).
PROJECTION_OUTPUT_DIM = 256

VALID_SPECIAL_RESIDUE_POLICIES = ("keep_if_supported", "error", "replace_with_X")
VALID_DTYPES = ("float32", "float16", "bfloat16")
VALID_DEVICES = ("auto", "cuda", "cpu")
VALID_AGGREGATION_METHODS = ("mean",)

DEFAULT_CANDIDATES = ["biohub/ESMC-600M-hf", "biohub/ESMC-600M"]


@dataclass
class ModelConfig:
    candidates: List[str] = field(default_factory=lambda: list(DEFAULT_CANDIDATES))
    frozen: bool = True
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    allow_masked_lm_fallback: bool = False
    allow_unresolved_revision: bool = False

    def validate(self) -> None:
        if not self.candidates:
            raise ConfigError("model.candidates must contain at least one checkpoint id")
        for candidate in self.candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                raise ConfigError(f"model.candidates entries must be non-empty strings, got {candidate!r}")
        if not self.frozen:
            raise ConfigError(
                "model.frozen must be true: this pipeline extracts label-free features with "
                "zero trainable parameters; ESMC is never fine-tuned here."
            )
        if self.trust_remote_code:
            # Allowed only by explicit opt-in, but loudly recorded downstream.
            pass


@dataclass
class SequenceConfig:
    max_model_positions: Any = "auto"
    chunk_size: Any = "auto"
    overlap: int = 256
    special_residue_policy: str = "keep_if_supported"

    def validate(self) -> None:
        if self.special_residue_policy not in VALID_SPECIAL_RESIDUE_POLICIES:
            raise ConfigError(
                f"sequence.special_residue_policy must be one of "
                f"{list(VALID_SPECIAL_RESIDUE_POLICIES)}, got {self.special_residue_policy!r}. "
                "A 'remove' policy is deliberately not implemented: deleting a residue shifts "
                "every downstream position and changes the sequence biologically."
            )
        if not isinstance(self.overlap, int) or self.overlap < 0:
            raise ConfigError(f"sequence.overlap must be a non-negative int, got {self.overlap!r}")


@dataclass
class ProjectionConfig:
    enabled: bool = False
    output_dim: int = PROJECTION_OUTPUT_DIM
    checkpoint: Optional[str] = None

    def validate(self) -> None:
        if self.output_dim != PROJECTION_OUTPUT_DIM:
            raise ConfigError(
                f"projection.output_dim must be {PROJECTION_OUTPUT_DIM} to match the "
                f"CancerCombo Target token width, got {self.output_dim!r}"
            )
        if self.enabled and not self.checkpoint:
            raise ConfigError(
                "projection.enabled is true but projection.checkpoint is null. This repository "
                "does not train projections: point projection.checkpoint at a genuinely trained "
                "checkpoint exported from CancerCombo, or set projection.enabled: false."
            )


@dataclass
class AggregationConfig:
    method: str = "mean"

    def validate(self) -> None:
        if self.method not in VALID_AGGREGATION_METHODS:
            raise ConfigError(
                f"aggregation.method must be one of {list(VALID_AGGREGATION_METHODS)}; got "
                f"{self.method!r}. No attention aggregator is built here (Section 11)."
            )


@dataclass
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "float32"
    batch_size: int = 1
    seed: int = 0

    def validate(self) -> None:
        if self.device not in VALID_DEVICES:
            raise ConfigError(f"runtime.device must be one of {list(VALID_DEVICES)}, got {self.device!r}")
        if self.dtype not in VALID_DTYPES:
            raise ConfigError(f"runtime.dtype must be one of {list(VALID_DTYPES)}, got {self.dtype!r}")
        if self.batch_size != 1:
            raise ConfigError(
                f"runtime.batch_size must be 1; arbitrary batch sizes are not supported in this "
                f"offline pipeline. Got batch_size={self.batch_size!r}"
            )


@dataclass
class CacheConfig:
    enabled: bool = True
    path: Optional[str] = None
    save_every: int = 100

    def validate(self) -> None:
        if not self.enabled:
            raise ConfigError(
                "cache.enabled must be true: the shared protein cache is part of the scientific "
                "reproducibility infrastructure and is mandatory."
            )
        if self.save_every < 1:
            raise ConfigError(f"cache.save_every must be >= 1, got {self.save_every!r}")


@dataclass
class OutputConfig:
    directory: str = "outputs/full"
    export_csv: bool = False
    diagnostics: bool = True
    diagnostics_max_pairs: int = 20000
    cosine_collapse_warn_threshold: float = 0.995
    norm_outlier_warn_factor: float = 10.0

    def validate(self) -> None:
        if self.diagnostics_max_pairs < 1:
            raise ConfigError("output.diagnostics_max_pairs must be >= 1")


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    source_path: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for section in (self.model, self.sequence, self.projection, self.aggregation,
                        self.runtime, self.cache, self.output):
            section.validate()
        if self.sequence.chunk_size not in (None, "auto"):
            if not isinstance(self.sequence.chunk_size, int) or self.sequence.chunk_size < 1:
                raise ConfigError(f"sequence.chunk_size must be 'auto' or a positive int, "
                                  f"got {self.sequence.chunk_size!r}")
        if self.sequence.max_model_positions not in (None, "auto"):
            if not isinstance(self.sequence.max_model_positions, int) or self.sequence.max_model_positions < 1:
                raise ConfigError("sequence.max_model_positions must be 'auto' or a positive int, "
                                  f"got {self.sequence.max_model_positions!r}")

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "model": asdict(self.model),
            "sequence": asdict(self.sequence),
            "projection": asdict(self.projection),
            "aggregation": asdict(self.aggregation),
            "runtime": asdict(self.runtime),
            "cache": asdict(self.cache),
            "output": asdict(self.output),
        }
        return copy.deepcopy(payload)

    def config_hash(self) -> str:
        """Hash of the *scientific* settings (excludes output plumbing)."""
        payload = self.to_dict()
        payload["output"] = {"directory": payload["output"]["directory"]}
        return sha256_json(payload)

    def apply_overrides(self, **kwargs: Any) -> "AppConfig":
        """Apply CLI overrides; returns self for chaining."""
        device = kwargs.get("device")
        if device:
            self.runtime.device = device
        if kwargs.get("projection"):
            self.projection.enabled = True
        if kwargs.get("export_csv"):
            self.output.export_csv = True
        output_dir = kwargs.get("output_dir")
        if output_dir:
            self.output.directory = output_dir
        cache_path = kwargs.get("cache_path")
        if cache_path:
            self.cache.path = cache_path
        self.validate()
        return self


_SECTIONS = {
    "model": ModelConfig,
    "sequence": SequenceConfig,
    "projection": ProjectionConfig,
    "aggregation": AggregationConfig,
    "runtime": RuntimeConfig,
    "cache": CacheConfig,
    "output": OutputConfig,
}


def _build_section(cls, payload: Any, name: str):
    if payload is None:
        return cls()
    if not isinstance(payload, dict):
        raise ConfigError(f"Config section '{name}' must be a mapping, got {type(payload).__name__}")
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = [k for k in payload if k not in known]
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in config section '{name}': {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    return cls(**payload)


def load_config(path: Optional[os.PathLike | str] = None) -> AppConfig:
    """Load and validate the YAML config. Missing file -> defaults."""
    raw: Dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config file {path} must contain a YAML mapping at the top level")
        raw = loaded

    seq_payload = dict(raw.get("sequence") or {})
    # Backwards-compatible alias (old key name).
    if "invalid_residue_policy" in seq_payload:
        if "special_residue_policy" in seq_payload:
            raise ConfigError(
                "Config sets both sequence.special_residue_policy and the deprecated alias "
                "sequence.invalid_residue_policy; set only sequence.special_residue_policy."
            )
        seq_payload["special_residue_policy"] = seq_payload.pop("invalid_residue_policy")

    sections = {
        "model": _build_section(ModelConfig, raw.get("model"), "model"),
        "sequence": _build_section(SequenceConfig, seq_payload, "sequence"),
        "projection": _build_section(ProjectionConfig, raw.get("projection"), "projection"),
        "aggregation": _build_section(AggregationConfig, raw.get("aggregation"), "aggregation"),
        "runtime": _build_section(RuntimeConfig, raw.get("runtime"), "runtime"),
        "cache": _build_section(CacheConfig, raw.get("cache"), "cache"),
        "output": _build_section(OutputConfig, raw.get("output"), "output"),
    }

    unknown_top = [k for k in raw if k not in _SECTIONS]
    if unknown_top:
        raise ConfigError(f"Unknown top-level config section(s): {sorted(unknown_top)}. "
                          f"Valid sections: {sorted(_SECTIONS)}")

    cfg = AppConfig(source_path=str(path) if path else None, raw=copy.deepcopy(raw), **sections)
    cfg.validate()
    return cfg
