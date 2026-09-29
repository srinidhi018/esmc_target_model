"""Error taxonomy.

FATAL vs ROW-LEVEL errors must never be confused.

* :class:`FatalError` aborts the whole run with a clear message. It must never be
  caught and funnelled into ``failed_rows.csv``.
* :class:`RowError` marks a single input row as ``status="failed"``; the run
  continues.
"""

from __future__ import annotations

from typing import Optional


class EsmcTargetError(Exception):
    """Base class for all errors raised by this package."""


class FatalError(EsmcTargetError):
    """Unrecoverable condition -> abort the run.

    Raised for: unresolvable/unverifiable ESMC checkpoint, hidden-size or
    architecture mismatch, startup token/residue alignment failure, cache
    fingerprint mismatch without ``--rebuild-cache``, corrupt cache, missing
    required columns, no authoritative model sequence capacity, unsupported
    aggregation method, projection enabled without a loadable checkpoint,
    subset-run isolation violation.
    """

    category = "fatal"


class ModelResolutionError(FatalError):
    """No candidate checkpoint resolved and verified as official ESMC-600M."""


class AlignmentError(FatalError):
    """Token <-> residue alignment invariant violated (startup self-check)."""


class ProvenanceError(FatalError):
    """Provenance tracking failure (e.g. unresolved attention backend or unverified revision)."""


class CapacityError(FatalError):
    """No authoritative maximum model sequence length could be established."""


class ConfigError(FatalError):
    """Configuration is invalid or internally inconsistent."""


class IsolationError(FatalError):
    """A subset run (``--max-rows`` / ``--nsc-ids``) would write to a full-run dir."""


class FingerprintMismatchError(FatalError):
    """Cached embeddings were produced under different settings."""


class CacheCorruptError(FatalError):
    """Cache file exists but cannot be parsed / is structurally invalid."""


class PipelineInterruptedError(FatalError):
    """Execution was interrupted by SIGINT / SIGTERM."""

    def __init__(self, message: str, shutdown_signal: Optional[str] = None,
                 processed_count: int = 0, total_count: int = 0) -> None:
        super().__init__(message)
        self.shutdown_signal = shutdown_signal or (message if message in ("SIGINT", "SIGTERM") else None)
        self.processed_count = processed_count
        self.total_count = total_count
        self.exit_code = 130 if self.shutdown_signal == "SIGINT" else 143


    @property
    def signal_name(self) -> Optional[str]:
        return self.shutdown_signal




class ProjectionError(FatalError):
    """Projection enabled but no loadable, genuinely trained checkpoint."""


class CudaOutOfMemoryError(FatalError):
    """CUDA OOM that could not be safely recovered.

    An OOM is *not* a biologically invalid row. It is only downgraded to a
    row-level failure when :func:`safe_recover_oom` has verified that the
    process is in a known-safe state.
    """

    def __init__(self, message: str, sequence_hash: Optional[str] = None,
                 sequence_length: Optional[int] = None) -> None:
        super().__init__(message)
        self.sequence_hash = sequence_hash
        self.sequence_length = sequence_length


class RowError(EsmcTargetError):
    """Row-level failure: recorded in ``error``; the run continues."""

    category = "row"

    def __init__(self, message: str, kind: str = "invalid_sequence") -> None:
        super().__init__(message)
        self.kind = kind


class InvalidSequenceError(RowError):
    def __init__(self, message: str) -> None:
        super().__init__(message, kind="invalid_sequence")


class UnsupportedResidueError(RowError):
    def __init__(self, message: str) -> None:
        super().__init__(message, kind="unsupported_residue")


class EmbeddingQualityError(RowError):
    def __init__(self, message: str) -> None:
        super().__init__(message, kind="embedding_quality")


class InferenceError(RowError):
    def __init__(self, message: str) -> None:
        super().__init__(message, kind="inference_error")
