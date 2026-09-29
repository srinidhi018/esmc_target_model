"""Cache behaviour: hit/miss, fingerprint refusal, atomic write, resume."""

from __future__ import annotations

import os

import pytest
import torch

from esmc_target.cache import (
    ProteinCache,
    build_fingerprint,
    fingerprint_diff,
)
from esmc_target.errors import CacheCorruptError, FingerprintMismatchError, FatalError
from esmc_target.sequence import sequence_hash


def fingerprint(**overrides):
    base = dict(
        cache_schema_version=1,
        model_id="biohub/ESMC-600M-hf",
        model_revision="rev1",
        tokenizer_revision="rev1",
        model_class="EsmcForMaskedLM",
        tokenizer_class="EsmcTokenizer",
        model_config_hash="mc",
        tokenizer_config_hash="tc",
        transformers_version="4.57.0",
        sequence_cleaning_version="clean-v1",
        special_residue_policy="keep_if_supported",
        special_residue_tokenizer_support={"U": True},
        chunk_size=2046,
        overlap=256,
        pooling_method="residue_mean",
        inference_dtype="float32",
        effective_attention_implementation="eager",
        pooling_implementation_version="pool-v1",
    )
    base.update(overrides)
    return build_fingerprint(**base)


def make_entry(value: float = 1.5):
    return torch.full((1152,), value, dtype=torch.float32)


def test_cache_miss_then_hit(tmp_path):
    cache = ProteinCache(tmp_path / "protein_cache.pt", fingerprint())
    digest = sequence_hash("ACDEFGHIKL")
    assert cache.get(digest) is None
    assert cache.stats.misses == 1
    cache.put(digest, make_entry(), ["P12345"], ["T1"], 10, 1,
              {"embedding_mean": 1.5, "embedding_std": 0.0, "embedding_l2_norm": 48.2})
    assert cache.get(digest) is not None
    assert cache.stats.hits == 1
    entry = cache.entry(digest)
    assert entry["uniprot_ids"] == ["P12345"] and entry["target_ids"] == ["T1"]
    assert entry["num_chunks"] == 1 and entry["sequence_length"] == 10
    assert entry["embedding_l2_norm"] == pytest.approx(48.2)


def test_provenance_lists_are_deduplicated_and_sorted(tmp_path):
    cache = ProteinCache(tmp_path / "c.pt", fingerprint())
    digest = sequence_hash("ACDEF")
    cache.put(digest, make_entry(), ["P2", "P1", "P2"], ["T2", "T1", "T2"], 5, 1, {})
    cache.put(digest, make_entry(), ["P3"], ["T0"], 5, 1, {})
    entry = cache.entry(digest)
    assert entry["uniprot_ids"] == ["P1", "P2", "P3"]
    assert entry["target_ids"] == ["T0", "T1", "T2"]


def test_provenance_does_not_affect_cache_identity(tmp_path):
    digest = sequence_hash("ACDEF")
    a = ProteinCache(tmp_path / "a.pt", fingerprint())
    b = ProteinCache(tmp_path / "b.pt", fingerprint())
    a.put(digest, make_entry(1.0), ["P1"], ["T1"], 5, 1, {})
    b.put(digest, make_entry(1.0), ["P9"], ["T9"], 5, 1, {})
    assert torch.equal(a.entries[digest]["embedding"], b.entries[digest]["embedding"])
    assert (a.fingerprint["fingerprint_hash"] == b.fingerprint["fingerprint_hash"])


def test_fingerprint_mismatch_refusal_lists_differing_fields(tmp_path):
    path = tmp_path / "protein_cache.pt"
    cache = ProteinCache(path, fingerprint(overlap=256))
    digest = sequence_hash("ACDEFGHIKL")
    cache.put(digest, make_entry(), [], [], 10, 1, {})
    cache.save(force=True)

    with pytest.raises(FingerprintMismatchError) as excinfo:
        ProteinCache(path, fingerprint(overlap=0))
    message = str(excinfo.value)
    assert "overlap" in message and "rebuild-cache" in message
    assert list(fingerprint_diff(fingerprint(overlap=256), fingerprint(overlap=0))) == ["overlap"]


def test_fingerprint_mismatch_refused_for_other_settings(tmp_path):
    path = tmp_path / "protein_cache.pt"
    ProteinCache(path, fingerprint(dtype="float32")).save(force=True)
    with pytest.raises(FingerprintMismatchError):
        ProteinCache(path, fingerprint(overlap=0, inference_dtype="bfloat16"))
    with pytest.raises(FingerprintMismatchError):
        ProteinCache(path, fingerprint(model_revision="other"))


def test_rebuild_cache_overrides_mismatch(tmp_path):
    path = tmp_path / "protein_cache.pt"
    cache = ProteinCache(path, fingerprint(overlap=256))
    cache.put(sequence_hash("ACDEF"), make_entry(), [], [], 5, 1, {})
    cache.save(force=True)
    rebuilt = ProteinCache(path, fingerprint(overlap=0), rebuild=True)
    assert len(rebuilt) == 0
    rebuilt.save(force=True)
    assert len(ProteinCache(path, fingerprint(overlap=0))) == 0


def test_atomic_write_leaves_no_partial_file(tmp_path):
    path = tmp_path / "protein_cache.pt"
    cache = ProteinCache(path, fingerprint())
    cache.put(sequence_hash("ACDEF"), make_entry(), [], [], 5, 1, {})
    cache.save(force=True)
    assert path.exists()
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], "temp files must be renamed away, not left behind"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "fingerprint" in payload and "entries" in payload


def test_atomic_write_removes_temp_file_on_failure(tmp_path):
    from esmc_target.utils import atomic_path

    path = tmp_path / "boom.json"
    with pytest.raises(RuntimeError):
        with atomic_path(path) as tmp:
            tmp.write_text("partial", encoding="utf-8")
            raise RuntimeError("simulated crash mid-write")
    assert not path.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_resume_skips_already_computed_hashes(tmp_path):
    path = tmp_path / "protein_cache.pt"
    first = ProteinCache(path, fingerprint())
    digest = sequence_hash("ACDEFGHIKL")
    first.put(digest, make_entry(2.0), [], [], 10, 1, {})
    first.save(force=True)

    second = ProteinCache(path, fingerprint())
    assert second.get(digest) is not None   # resume: hit
    assert second.stats.misses == 0
    assert len(second) == 1


def test_save_every_controls_write_frequency(tmp_path):
    path = tmp_path / "protein_cache.pt"
    cache = ProteinCache(path, fingerprint())
    for i in range(9):
        cache.put(sequence_hash(f"SEQ{i}"), make_entry(), [], [], 3, 1, {})
    assert cache.save(save_every=10) is False
    cache.put(sequence_hash("SEQ9"), make_entry(), [], [], 3, 1, {})
    assert cache.save(save_every=10) is True


def test_corrupt_cache_is_fatal_not_silently_ignored(tmp_path):
    path = tmp_path / "protein_cache.pt"
    path.write_bytes(b"not a torch file")
    with pytest.raises(CacheCorruptError):
        ProteinCache(path, fingerprint())


def test_cache_missing_keys_is_fatal(tmp_path):
    path = tmp_path / "protein_cache.pt"
    torch.save({"nothing": 1}, path)
    with pytest.raises(CacheCorruptError):
        ProteinCache(path, fingerprint())


def test_fingerprint_requires_all_fields():
    with pytest.raises(FatalError):
        build_fingerprint(model_id="x")
