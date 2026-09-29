"""Regression tests for ESMC checkpoint provenance and tokenizer revision resolution.

Covers:
A. Model commit known + tokenizer object has _commit_hash=None + tokenizer files
   resolved from same immutable repository revision => provenance passes.
B. Tokenizer independently resolved to a different revision => provenance records
   the different revision and does NOT collapse it to model_rev.
C. Tokenizer revision genuinely unresolved => provenance fails with ModelResolutionError.
D. allow_unresolved_revision=True remains an explicit escape hatch and is False
   in normal production configuration.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from conftest import MockEsmcConfig, MockTokenizer
from esmc_target.config import load_config
from esmc_target.errors import ModelResolutionError
from esmc_target.esmc_encoder import (
    _extract_commit_hash_from_path,
    _resolve_tokenizer_revision,
    resolve_model,
)


def _setup_mock_loaders(monkeypatch, model_rev, tok_rev_path=None, tok_explicit_sha=None):
    """Helper to mock load_model_and_tokenizer and checkpoint verification."""
    config = MockEsmcConfig()
    config._commit_hash = model_rev

    model = MagicMock()
    model.config = config

    tokenizer = MockTokenizer()
    tokenizer._commit_hash = tok_explicit_sha

    monkeypatch.setattr(
        "esmc_target.esmc_encoder.load_model_and_tokenizer",
        lambda candidate, **kwargs: (model, tokenizer, "MockEsmcModel", "MockTokenizer", "mock_api", False)
    )
    monkeypatch.setattr(
        "esmc_target.esmc_encoder._verify_checkpoint",
        lambda model_config, candidate: (True, {"owner_is_official_publisher": True}, None)
    )

    if tok_rev_path is not None:
        monkeypatch.setattr(
            "huggingface_hub.try_to_load_from_cache",
            lambda repo_id, filename, **kwargs: tok_rev_path
        )
    else:
        monkeypatch.setattr(
            "huggingface_hub.try_to_load_from_cache",
            lambda repo_id, filename, **kwargs: None
        )

    # Disable remote HfApi lookup in isolated unit tests
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda self, repo_id, **kwargs: MagicMock(sha=None)
    )

    return model, tokenizer


def test_regression_a_model_known_tok_none_cache_same_revision(monkeypatch):
    """A. Model commit known + tokenizer object has _commit_hash=None + tokenizer

    files resolved from same immutable repository revision => provenance passes.
    """
    shared_sha = "0fb34e7e5fe1f85d0abaa3d35e2671107c0b458c"
    cache_file = f"/root/.cache/huggingface/hub/models--biohub--ESMC-600M-hf/snapshots/{shared_sha}/tokenizer.json"

    _setup_mock_loaders(monkeypatch, model_rev=shared_sha, tok_rev_path=cache_file, tok_explicit_sha=None)

    resolved = resolve_model(["biohub/ESMC-600M-hf"], allow_unresolved_revision=False)
    assert resolved.model_revision == shared_sha
    assert resolved.tokenizer_revision == shared_sha
    assert resolved.revision_resolved is True
    assert resolved.provenance_complete is True


def test_regression_b_tok_independent_different_revision(monkeypatch):
    """B. Tokenizer independently resolved to a different revision => provenance

    records the different revision and does NOT collapse it to model_rev.
    """
    model_sha = "0fb34e7e5fe1f85d0abaa3d35e2671107c0b458c"
    tok_sha = "1111222233334444555566667777888899990000"
    cache_file = f"/root/.cache/huggingface/hub/models--biohub--ESMC-600M-hf/snapshots/{tok_sha}/tokenizer.json"

    _setup_mock_loaders(monkeypatch, model_rev=model_sha, tok_rev_path=cache_file, tok_explicit_sha=None)

    resolved = resolve_model(["biohub/ESMC-600M-hf"], allow_unresolved_revision=False)
    assert resolved.model_revision == model_sha
    assert resolved.tokenizer_revision == tok_sha
    assert resolved.tokenizer_revision != resolved.model_revision
    assert resolved.revision_resolved is True
    assert resolved.provenance_complete is True


def test_regression_c_tok_genuinely_unresolved_fails(monkeypatch):
    """C. Tokenizer revision genuinely unresolved => provenance fails."""
    model_sha = "0fb34e7e5fe1f85d0abaa3d35e2671107c0b458c"

    _setup_mock_loaders(monkeypatch, model_rev=model_sha, tok_rev_path=None, tok_explicit_sha=None)

    with pytest.raises(ModelResolutionError, match="Could not resolve exact commit revision SHA"):
        resolve_model(["biohub/ESMC-600M-hf"], allow_unresolved_revision=False)


def test_regression_d_allow_unresolved_revision_escape_hatch(monkeypatch):
    """D. allow_unresolved_revision=True remains an explicit escape hatch and is

    NOT used by normal production config.
    """
    # Part 1: Verify production configuration defaults allow_unresolved_revision to False
    config = load_config(None)
    assert config.model.allow_unresolved_revision is False

    # Also inspect configs/config.yaml on disk directly
    config_disk = load_config(Path(__file__).resolve().parents[1] / "configs" / "config.yaml")
    assert config_disk.model.allow_unresolved_revision is False

    # Part 2: Verify escape hatch allows execution when explicitly enabled
    model_sha = "0fb34e7e5fe1f85d0abaa3d35e2671107c0b458c"
    _setup_mock_loaders(monkeypatch, model_rev=model_sha, tok_rev_path=None, tok_explicit_sha=None)

    resolved = resolve_model(["biohub/ESMC-600M-hf"], allow_unresolved_revision=True)
    assert resolved.model_revision == model_sha
    assert resolved.tokenizer_revision == "unresolved"
    assert resolved.revision_resolved is False


def test_extract_commit_hash_from_path_posix_and_windows():
    """Verify path extraction works with both Windows backslashes and POSIX slashes."""
    sha = "0fb34e7e5fe1f85d0abaa3d35e2671107c0b458c"
    posix_path = f"/root/.cache/huggingface/hub/models--biohub--ESMC-600M-hf/snapshots/{sha}/tokenizer.json"
    win_path = rf"C:\Users\User\.cache\huggingface\hub\models--biohub--ESMC-600M-hf\snapshots\{sha}\tokenizer.json"

    assert _extract_commit_hash_from_path(posix_path) == sha
    assert _extract_commit_hash_from_path(win_path) == sha
    assert _extract_commit_hash_from_path("/invalid/path/without/snapshot") is None
    assert _extract_commit_hash_from_path("") is None
