"""Opt-in integration test that REALLY downloads and runs ESMC-600M.

Skipped by default so the default test run needs no network and no ~2.4 GB
download. Enable it explicitly:

    ESMC_REAL_RUN=1 pytest tests/test_integration_esmc.py -v

This test is the only place in the repository where a real ESMC forward pass
occurs; every other test uses the mock in ``conftest.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from esmc_target.config import load_config  # noqa: E402
from esmc_target.errors import FatalError  # noqa: E402
from esmc_target.esmc_encoder import (  # noqa: E402
    TargetProteinEncoder,
    inspect_runtime,
    probe_special_residue_support,
    resolve_model,
)
from esmc_target.pipeline import resolve_device  # noqa: E402
from esmc_target.pooling import validate_embedding  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("ESMC_REAL_RUN") != "1",
    reason="set ESMC_REAL_RUN=1 to download and run the real ESMC-600M checkpoint",
)


def test_real_esmc_pipeline_end_to_end(tmp_path):
    config = load_config(None)
    device = resolve_device(config.runtime.device)
    try:
        resolved = resolve_model(config.model.candidates,
                                 trust_remote_code=config.model.trust_remote_code)
    except FatalError as exc:
        pytest.skip(f"ESMC-600M could not be resolved/verified here: {exc}")

    assert resolved.hidden_size == 1152
    assert resolved.model_max_positions > 0

    encoder = TargetProteinEncoder(resolved=resolved, device=device,
                                   dtype=config.runtime.dtype,
                                   chunk_size="auto", overlap=config.sequence.overlap)
    inspection = inspect_runtime(resolved.model, resolved.tokenizer, device=device,
                                 dtype=encoder.dtype)
    assert inspection.tokenization.residue_tokens == list("ACDEFGHIKL")
    assert len(inspection.tokenization.residue_token_indices) == 10
    assert inspection.special_tokens >= 1

    support = probe_special_residue_support(resolved.model, resolved.tokenizer, device=device,
                                            dtype=encoder.dtype)
    for symbol, result in support.items():
        assert result.primary_criterion_passed, f"{symbol}: {result.detail}"
        print(f"{symbol}: supported={result.supported} ({result.detail})")

    out = encoder.encode("ACDEFGHIKL")
    assert out.embedding.shape == (1152,)
    assert out.embedding.dtype == torch.float32
    validate_embedding(out.embedding, 1152)

    # a long sequence must be chunked with full coverage
    long_sequence = "".join("ACDEFGHIKLMNPQRSTVWY"[(i * 7) % 20] for i in range(encoder.residue_capacity + 50))
    long_out = encoder.encode(long_sequence)
    assert long_out.num_chunks > 1
    assert long_out.coverage["min_coverage_count"] >= 1
    assert long_out.coverage["num_residues_pooled"] == len(long_sequence)

    # determinism: the same sequence encodes identically in-process
    again = encoder.encode("ACDEFGHIKL")
    assert torch.allclose(out.embedding, again.embedding, atol=1e-6)
    print("model:", resolved.model_id, resolved.model_revision, resolved.architecture)
    print("capacity:", encoder.model_max_positions, encoder.special_tokens, encoder.residue_capacity)
