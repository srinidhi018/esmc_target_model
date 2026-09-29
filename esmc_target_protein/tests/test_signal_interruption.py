"""Unit tests for signal interruption, manifest recording, and resume semantics (Tests AP - AT)."""

import pytest
import signal
from esmc_target.signals import SignalHandler
from esmc_target.errors import PipelineInterruptedError


def test_signal_handler_sets_flags_only():
    # Test AP & AQ: Signal handler sets boolean flag only, no heavy work
    handler = SignalHandler()
    assert handler.shutdown_requested is False
    assert handler.shutdown_signal is None

    # Simulate SIGINT signal call
    handler._handle(signal.SIGINT, None)
    assert handler.shutdown_requested is True
    assert handler.shutdown_signal == "SIGINT"

    # Reset
    handler.shutdown_requested = False
    handler.shutdown_signal = None

    # Simulate SIGTERM signal call
    handler._handle(signal.SIGTERM, None)
    assert handler.shutdown_requested is True
    assert handler.shutdown_signal == "SIGTERM"


def test_interrupted_error_attributes():
    # Test AR: Interrupted error payload attributes for manifest recording
    err = PipelineInterruptedError("SIGINT", processed_count=5, total_count=10)
    assert err.signal_name == "SIGINT"
    assert err.processed_count == 5
    assert err.total_count == 10
    assert err.exit_code in (130, 143)
