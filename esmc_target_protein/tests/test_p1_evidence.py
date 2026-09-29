"""Verification suite for P1 Patches (Items 1 - 5, plus Attention Provenance Hardening)."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch

from esmc_target.cache import build_fingerprint
from esmc_target.errors import FatalError, ModelResolutionError, ProvenanceError
from esmc_target.esmc_encoder import (
    ResolvedModel,
    _is_candidate_not_found_error,
    check_applies_final_layer_norm,
    determine_attention_implementation,
    resolve_model,
)
from esmc_target.pipeline import PreprocessingPipeline
from conftest import make_resolved


def test_item1_check_applies_final_layer_norm():
    class ModuleWithNorm:
        norm = torch.nn.Identity()

    class ModuleWithoutNorm:
        pass

    class ConfigWithNormFlag:
        final_layernorm = True

    class ConfigWithoutNormFlag:
        final_layernorm = False

    # Case 1: model.norm exists -> True
    assert check_applies_final_layer_norm(ModuleWithNorm(), ConfigWithoutNormFlag()) is True

    # Case 2: final_layernorm flag is True -> True
    assert check_applies_final_layer_norm(ModuleWithoutNorm(), ConfigWithNormFlag()) is True

    # Case 3: neither -> False
    assert check_applies_final_layer_norm(ModuleWithoutNorm(), ConfigWithoutNormFlag()) is False


def test_item2_pipeline_config_plumbing(monkeypatch, config):
    received_kwargs = {}

    def spy_resolve_model(candidates, **kwargs):
        nonlocal received_kwargs
        received_kwargs = kwargs
        return make_resolved()

    monkeypatch.setattr("esmc_target.pipeline.resolve_model", spy_resolve_model)

    pipeline = PreprocessingPipeline(config=config, output_dir=Path("/tmp/test"), input_path=None)

    # Test True / True
    pipeline.config.model.allow_masked_lm_fallback = True
    pipeline.config.model.allow_unresolved_revision = True
    pipeline._resolve_model()

    assert received_kwargs.get("allow_masked_lm_fallback") is True
    assert received_kwargs.get("allow_unresolved_revision") is True

    # Test False / False
    pipeline.config.model.allow_masked_lm_fallback = False
    pipeline.config.model.allow_unresolved_revision = False
    pipeline._resolve_model()

    assert received_kwargs.get("allow_masked_lm_fallback") is False
    assert received_kwargs.get("allow_unresolved_revision") is False


def test_item3_is_candidate_not_found_error_classification():
    class DummyHTTPError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code
            super().__init__(f"HTTP Error {status_code}")

    class GatedRepoError(Exception):
        pass

    class RepositoryNotFoundError(Exception):
        pass

    # 401 -> True
    assert _is_candidate_not_found_error(DummyHTTPError(401)) is True
    # 404 -> True
    assert _is_candidate_not_found_error(DummyHTTPError(404)) is True
    # GatedRepoError -> True
    assert _is_candidate_not_found_error(GatedRepoError("gated repository")) is True
    # RepositoryNotFoundError -> True
    assert _is_candidate_not_found_error(RepositoryNotFoundError("repo not found")) is True

    # 500 -> False
    assert _is_candidate_not_found_error(DummyHTTPError(500)) is False
    # 503 -> False
    assert _is_candidate_not_found_error(DummyHTTPError(503)) is False
    # TimeoutError -> False
    assert _is_candidate_not_found_error(TimeoutError("connection timed out")) is False
    # ConnectionResetError -> False
    assert _is_candidate_not_found_error(ConnectionResetError("connection reset by peer")) is False


def test_item4_lazy_transformers_import_missing_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ModelResolutionError) as exc_info:
        resolve_model(["biohub/ESMC-600M-hf"])

    assert "transformers is not importable" in str(exc_info.value) or "REQUIRES native Transformers" in str(exc_info.value)


def test_item5_attention_implementation_provenance_and_fingerprint():
    class ModelWithAttnImpl:
        config = type("Config", (), {"_attn_implementation": "sdpa"})()

    class ModelWithNoAttnImpl:
        config = type("Config", (), {})()

    # Determine backend correctly
    assert determine_attention_implementation(ModelWithAttnImpl(), ModelWithAttnImpl.config) == "sdpa"

    # Raise ProvenanceError when undetermined
    with pytest.raises(ProvenanceError):
        determine_attention_implementation(ModelWithNoAttnImpl(), ModelWithNoAttnImpl.config)

    # Fingerprint diff test
    fp_eager = build_fingerprint(
        cache_schema_version=1,
        model_id="biohub/ESMC-600M-hf",
        model_revision="rev1",
        model_class="EsmcModel",
        tokenizer_class="EsmcTokenizer",
        model_config_hash="cfg1",
        tokenizer_config_hash="tok1",
        sequence_cleaning_version="v1",
        special_residue_policy="keep_if_supported",
        special_residue_tokenizer_support={},
        chunk_size=2046,
        overlap=256,
        pooling_method="residue_mean",
        inference_dtype="float32",
        pooling_implementation_version="pool-v1",
        effective_attention_implementation="eager",
    )

    fp_sdpa = build_fingerprint(
        cache_schema_version=1,
        model_id="biohub/ESMC-600M-hf",
        model_revision="rev1",
        model_class="EsmcModel",
        tokenizer_class="EsmcTokenizer",
        model_config_hash="cfg1",
        tokenizer_config_hash="tok1",
        sequence_cleaning_version="v1",
        special_residue_policy="keep_if_supported",
        special_residue_tokenizer_support={},
        chunk_size=2046,
        overlap=256,
        pooling_method="residue_mean",
        inference_dtype="float32",
        pooling_implementation_version="pool-v1",
        effective_attention_implementation="sdpa",
    )

    assert fp_eager["fingerprint_hash"] != fp_sdpa["fingerprint_hash"]
    assert fp_eager["effective_attention_implementation"] == "eager"
    assert fp_sdpa["effective_attention_implementation"] == "sdpa"


def test_hole2_cache_fingerprint_missing_attention_implementation_raises():
    with pytest.raises(FatalError) as exc_info:
        build_fingerprint(
            cache_schema_version=1,
            model_id="biohub/ESMC-600M-hf",
            model_revision="rev1",
            model_class="EsmcModel",
            tokenizer_class="EsmcTokenizer",
            model_config_hash="cfg1",
            tokenizer_config_hash="tok1",
            sequence_cleaning_version="v1",
            special_residue_policy="keep_if_supported",
            special_residue_tokenizer_support={},
            chunk_size=2046,
            overlap=256,
            pooling_method="residue_mean",
            inference_dtype="float32",
            pooling_implementation_version="pool-v1",
            # effective_attention_implementation omitted!
        )
    assert "missing required field(s)" in str(exc_info.value)
    assert "effective_attention_implementation" in str(exc_info.value)


def test_hole3_no_classname_fallback_in_determine_attention_implementation():
    class DummyAttentionModule:
        pass

    class ModelWithAttentionModuleClassOnly:
        layers = [type("Layer", (), {"self_attn": DummyAttentionModule()})()]

    config = type("Config", (), {})()

    with pytest.raises(ProvenanceError) as exc_info:
        determine_attention_implementation(ModelWithAttentionModuleClassOnly(), config)

    assert "Could not determine the effective attention implementation" in str(exc_info.value)


def test_hole4_resolved_model_to_dict_includes_effective_attention_implementation():
    res = make_resolved()
    dict_repr = res.to_dict()
    assert "effective_attention_implementation" in dict_repr
    assert dict_repr["effective_attention_implementation"] == "eager"


def _make_dummy_resolved(**overrides):
    base = dict(
        model_id="biohub/ESMC-600M-hf",
        model_revision="rev1",
        model=None,
        tokenizer=None,
        model_class="EsmcModel",
        tokenizer_class="EsmcTokenizer",
        architecture="EsmcForMaskedLM",
        hidden_size=1152,
        model_max_positions=2048,
        model_config=None,
        model_config_hash="cfg1",
        tokenizer_config_hash="tok1",
        transformers_version="5.0.0",
        api_used="mock",
        owner="biohub",
        applies_final_layer_norm=True,
        effective_attention_implementation="eager",
    )
    base.update(overrides)
    return ResolvedModel(**base)


def test_resolved_model_construction_guard_tightened():
    # 1. Empty string -> ProvenanceError
    with pytest.raises(ProvenanceError, match="must be a non-empty string"):
        _make_dummy_resolved(effective_attention_implementation="")

    # 2. None -> ProvenanceError
    with pytest.raises(ProvenanceError, match="must be a non-empty string"):
        _make_dummy_resolved(effective_attention_implementation=None)

    # 3. Whitespace string -> ProvenanceError
    with pytest.raises(ProvenanceError, match="must be a non-empty string"):
        _make_dummy_resolved(effective_attention_implementation="   ")

    # 4. Non-string integer 123 -> ProvenanceError
    with pytest.raises(ProvenanceError, match="must be a non-empty string"):
        _make_dummy_resolved(effective_attention_implementation=123)

    # 5. Uppercase "EAGER" -> successfully normalized to "eager"
    res_eager = _make_dummy_resolved(effective_attention_implementation="EAGER")
    assert res_eager.effective_attention_implementation == "eager"

    # 6. Lowercase "sdpa" -> successfully stored as "sdpa"
    res_sdpa = _make_dummy_resolved(effective_attention_implementation="sdpa")
    assert res_sdpa.effective_attention_implementation == "sdpa"


def test_resolved_model_omitting_effective_attention_implementation_raises_typeerror():
    # Expected TypeError because effective_attention_implementation intentionally has no default value
    kwargs = dict(
        model_id="biohub/ESMC-600M-hf",
        model_revision="rev1",
        model=None,
        tokenizer=None,
        model_class="EsmcModel",
        tokenizer_class="EsmcTokenizer",
        architecture="EsmcForMaskedLM",
        hidden_size=1152,
        model_max_positions=2048,
        model_config=None,
        model_config_hash="cfg1",
        tokenizer_config_hash="tok1",
        transformers_version="5.0.0",
        api_used="mock",
        owner="biohub",
        applies_final_layer_norm=True,
    )
    with pytest.raises(TypeError):
        ResolvedModel(**kwargs)
