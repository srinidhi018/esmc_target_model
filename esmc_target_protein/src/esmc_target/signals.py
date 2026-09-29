"""Signal management for graceful shutdown on SIGINT / SIGTERM.

Rules (Section 22 of specification):
- The signal handler MUST ONLY set a boolean flag (`shutdown_requested = True`)
  and record the signal name (`shutdown_signal = signal_name`).
- The signal handler MUST NOT: save files, serialize tensors, call CUDA, call the
  model, manipulate the cache, or perform heavy work.
- Main processing loop checks `shutdown_requested` after a safe completed unit
  (one complete unique protein encoding).
"""

from __future__ import annotations

import signal
import sys
from typing import Optional

from .utils import get_logger

LOGGER = get_logger("esmc_target.signals")


class SignalHandler:
    """Register SIGINT and SIGTERM handlers to flag graceful shutdown."""

    def __init__(self) -> None:
        self.shutdown_requested: bool = False
        self.shutdown_signal: Optional[str] = None
        self._old_sigint = None
        self._old_sigterm = None

    def register(self) -> None:
        """Attach signal handlers."""
        self._old_sigint = signal.signal(signal.SIGINT, self._handle)
        if hasattr(signal, "SIGTERM"):
            self._old_sigterm = signal.signal(signal.SIGTERM, self._handle)

    def restore(self) -> None:
        """Restore default signal handlers."""
        if self._old_sigint is not None:
            signal.signal(signal.SIGINT, self._old_sigint)
        if self._old_sigterm is not None and hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._old_sigterm)

    def _handle(self, signum: int, frame: object) -> None:
        name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        if not self.shutdown_requested:
            self.shutdown_requested = True
            self.shutdown_signal = name
            sys.stderr.write(f"\nReceived {name}. Finshing current protein and shutting down gracefully...\n")
        else:
            sys.stderr.write(f"\nReceived second {name}. Forcing exit.\n")
            sys.exit(128 + signum)
