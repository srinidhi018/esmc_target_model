"""Unit tests for cache fingerprinting, atomic save, inter-process lock, and branch metadata (Tests AH - AO)."""

import pytest
import torch
from esmc_target.cache import (
    ProteinCache,
    build_fingerprint,
    FingerprintMismatchError,
)


def get_dummy_fingerprint(extra="v1"):
    return build_fingerprint(
        cache_schema_version=1,
        model_id=f"esmc-600m-{extra}",
        model_revision="main",
        tokenizer_revision="main",
        model_class="EsmcModel",
        tokenizer_class="EsmcTokenizer",
        model_config_hash="abc",
        tokenizer_config_hash="def",
        transformers_version="5.0.0",
        sequence_cleaning_version="clean-v1",
        special_residue_policy="keep_if_supported",
        special_residue_tokenizer_support={"U": True},
        chunk_size=1000,
        overlap=100,
        pooling_method="mean",
        inference_dtype="float32",
        effective_attention_implementation="eager",
        pooling_implementation_version="pool-v1",
    )


def test_cache_fingerprint_match_and_mismatch(tmp_path):
    fp1 = get_dummy_fingerprint("v1")
    fp2 = get_dummy_fingerprint("v2")

    cache_file = tmp_path / "protein_cache.pt"
    cache = ProteinCache(cache_file, fp1)
    
    vec = torch.tensor([1.0, 2.0, 3.0])
    cache.put("seq_hash_1", vec, uniprot_ids=["P1"], target_ids=["T1"],
              sequence_length=10, num_chunks=1, diagnostics={"embedding_l2_norm": 3.74})
    cache.save(force=True)

    # Test AI: Loading with identical fingerprint succeeds
    cache_reload = ProteinCache(cache_file, fp1)
    assert "seq_hash_1" in cache_reload
    assert torch.allclose(cache_reload.get("seq_hash_1"), vec)

    # Test AH: Loading with mismatched fingerprint fails cleanly without overwriting
    with pytest.raises(FingerprintMismatchError, match="Differing fields"):
        ProteinCache(cache_file, fp2)


def test_cache_atomic_save_and_lock(tmp_path):
    # Test AJ & AK: Atomic save creates valid file and uses lock file
    fp = get_dummy_fingerprint()
    cache_file = tmp_path / "protein_cache.pt"
    cache = ProteinCache(cache_file, fp)
    
    cache.put("seq_1", torch.tensor([0.5, 0.5]), uniprot_ids=[], target_ids=[],
              sequence_length=5, num_chunks=1, diagnostics={})
    saved = cache.save(force=True)
    assert saved is True
    assert cache_file.exists()
    
    # Test lock acquisition
    with cache.lock:
        assert cache.lock_path.exists()



def test_cache_branch_metadata_same_and_cross_branch_hits(tmp_path):
    # Test AM, AN, AO: Branch provenance and accounting
    fp = get_dummy_fingerprint()
    cache_file = tmp_path / "protein_cache.pt"
    cache = ProteinCache(cache_file, fp)

    vec = torch.tensor([1.0, 1.0])
    cache.put("seq_1", vec, uniprot_ids=["P12345"], target_ids=["T1"],
              sequence_length=10, num_chunks=1, diagnostics={}, branch="target")
    
    entry = cache.entry("seq_1")
    assert entry["branches"] == ["target"]

    # Target same-branch hit
    v_hit1 = cache.get("seq_1", branch="target")
    assert v_hit1 is not None
    assert cache.stats.same_branch_cache_hits == 1
    assert cache.stats.cross_branch_cache_hits == 0

    # Pathway cross-branch hit (simulating future pathway reuse)
    v_hit2 = cache.get("seq_1", branch="pathway")
    assert v_hit2 is not None
    assert cache.stats.same_branch_cache_hits == 1
    assert cache.stats.cross_branch_cache_hits == 1

    updated_entry = cache.entry("seq_1")
    assert updated_entry["branches"] == ["pathway", "target"]  # Sorted list
