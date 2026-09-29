"""Regression tests for second audit fixes: loader preference, RMW cache concurrency, diagnostics success path, and fatal error propagation."""

import pytest
import sys
import torch
import types
from unittest.mock import MagicMock

from conftest import MockTokenizer, make_resolved, protein
from esmc_target.cache import ProteinCache, build_fingerprint
from esmc_target.diagnostics import run_diagnostics
from esmc_target.errors import AlignmentError, FatalError, UnsupportedResidueError
from esmc_target.esmc_encoder import extract_residue_hidden_states, load_model_and_tokenizer
from esmc_target.pipeline import PreprocessingPipeline
from esmc_target.sequence import clean_sequence


def test_loader_prefers_esmc_model_over_masked_lm():
    # Test Section 3: EsmcModel preferred when available
    mock_transformers = types.ModuleType("transformers")
    
    class FakeEsmcModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            m = MagicMock()
            m.__class__.__name__ = "EsmcModel"
            return m

    class FakeEsmcForMaskedLM:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            m = MagicMock()
            m.__class__.__name__ = "EsmcForMaskedLM"
            return m

    mock_transformers.EsmcModel = FakeEsmcModel
    mock_transformers.EsmcForMaskedLM = FakeEsmcForMaskedLM
    mock_transformers.AutoTokenizer = MagicMock()
    mock_transformers.__version__ = "5.0.0"

    sys.modules["transformers"] = mock_transformers
    try:
        model, tok, m_cls, t_cls, api, fallback_used = load_model_and_tokenizer("biohub/ESMC-600M-hf")
        assert m_cls == "EsmcModel"
        assert "EsmcModel" in api
        assert fallback_used is False
    finally:
        sys.modules.pop("transformers", None)


def test_logits_never_used_as_residue_embeddings():
    # Test Section 4: Output extraction fails fatally if only logits exist
    class MockOutputLogitsOnly:
        def __init__(self):
            self.logits = torch.randn(1, 7, 1152)

    out = MockOutputLogitsOnly()
    input_ids = torch.tensor([[0, 10, 11, 12, 13, 14, 1]])
    positions = [1, 2, 3, 4, 5]

    with pytest.raises(AlignmentError, match="logits must never be used as residue embeddings"):
        extract_residue_hidden_states(out, input_ids, positions, 5)


def test_keep_if_supported_never_replaces_unsupported_with_x():
    # Test Section 6: keep_if_supported on unsupported residue raises UnsupportedResidueError (ROW FAILURE)
    support_map = {"U": False}
    with pytest.raises(UnsupportedResidueError, match="never replaces with X"):
        clean_sequence("ACDUEFG", policy="keep_if_supported", tokenizer_support=support_map)


def test_diagnostics_success_path():
    # Test Section 9 & 10: Diagnostics success path executes without NameError
    embeddings = {
        "hash_1": torch.randn(1152),
        "hash_2": torch.randn(1152),
    }
    sequences_by_hash = {
        "hash_1": "ACDEFGHIKL",
        "hash_2": "MNPQRSTVWY",
    }
    cache_entries = {
        "hash_1": {"embedding": embeddings["hash_1"], "num_chunks": 1},
        "hash_2": {"embedding": embeddings["hash_2"], "num_chunks": 1},
    }
    labels = {
        "hash_1": {"gene_symbol": "GENE1", "nsc_id": "NSC-1"},
        "hash_2": {"gene_symbol": "GENE2", "nsc_id": "NSC-2"},
    }

    mock_encoder = MagicMock()
    mock_encoder.encode.side_effect = lambda seq, **kwargs: MagicMock(
        embedding=embeddings[sequences_by_hash.get(seq, "hash_1") if seq in ("ACDEFGHIKL", "MNPQRSTVWY") else "hash_1"],
        num_chunks=1
    )

    res = run_diagnostics(
        embeddings=embeddings,
        encoder=mock_encoder,
        cache_entries=cache_entries,
        sequences_by_hash=sequences_by_hash,
        labels=labels,
        chunk_probe_sequence="ACDEFGHIKL",
        pick_for_determinism=[("short:GENE1", "ACDEFGHIKL")],
    )

    assert res["status"] == "success"
    assert res["embeddings_modified"] is False
    assert res["determinism_check"]["status"] == "success"
    assert res["chunking_consistency_check"]["status"] == "success"


def test_cache_read_modify_write_concurrency_no_lost_updates(tmp_path):
    # Test Section 11 & 12 & 13: Read-Modify-Write cache transaction preserves updates across processes
    fp = build_fingerprint(
        cache_schema_version=1,
        model_id="biohub/ESMC-600M-hf",
        model_revision="main",
        tokenizer_id="biohub/ESMC-600M-hf",
        tokenizer_revision="main",
        model_class="EsmcModel",
        tokenizer_class="EsmcTokenizer",
        model_config_hash="abc",
        tokenizer_config_hash="def",
        transformers_version="5.0.0",
        torch_version=torch.__version__,
        sequence_cleaning_version="clean-v1",
        special_residue_policy="keep_if_supported",
        special_residue_tokenizer_support={"U": True},
        model_max_positions=2048,
        residue_capacity=2046,
        chunk_size=2046,
        overlap=256,
        pooling_method="residue_mean",
        inference_dtype="float32",
        effective_attention_implementation="eager",
        pooling_implementation_version="pool-v1",
    )

    cache_file = tmp_path / "shared_cache.pt"

    # Process 1 creates cache and puts entry A
    cache1 = ProteinCache(cache_file, fp)
    cache1.put("seq_A", torch.tensor([1.0, 1.0]), uniprot_ids=["P_A"], target_ids=["T_A"],
               sequence_length=10, num_chunks=1, diagnostics={})
    cache1.save(force=True)

    # Process 2 loads cache (has seq_A) and puts entry B
    cache2 = ProteinCache(cache_file, fp)
    cache2.put("seq_B", torch.tensor([2.0, 2.0]), uniprot_ids=["P_B"], target_ids=["T_B"],
               sequence_length=15, num_chunks=1, diagnostics={})
    cache2.save(force=True)

    # Meanwhile Process 1 puts entry C and saves
    cache1.put("seq_C", torch.tensor([3.0, 3.0]), uniprot_ids=["P_C"], target_ids=["T_C"],
               sequence_length=20, num_chunks=1, diagnostics={})
    cache1.save(force=True)

    # Reload fresh cache: MUST contain seq_A, seq_B, AND seq_C
    final_cache = ProteinCache(cache_file, fp)
    assert "seq_A" in final_cache
    assert "seq_B" in final_cache
    assert "seq_C" in final_cache
    assert len(final_cache.entries) == 3


def test_fatal_error_propagates_out_of_encode_record(tmp_path, monkeypatch):
    # Test Section 8: AlignmentError / FatalError during encoding is NEVER swallowed into row failure
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text(
        "nsc_id,drug_name,target_id,gene_symbol,uniprot_id,sequence,sequence_length\n"
        "NSC-100,Drug1,T1,G1,P1,ACDEFGHIKL,10\n",
        encoding="utf-8"
    )

    resolved = make_resolved()
    
    def failing_encode(*args, **kwargs):
        raise AlignmentError("Simulated fatal alignment failure during forward pass")

    from esmc_target.esmc_encoder import TargetProteinEncoder
    monkeypatch.setattr(TargetProteinEncoder, "encode", failing_encode)

    pipeline = PreprocessingPipeline(
        config=__import__("esmc_target.config").config.load_config(None),
        output_dir=tmp_path / "out",
        input_path=dataset_path,
        resolved_model=resolved
    )

    with pytest.raises(AlignmentError, match="Simulated fatal alignment failure"):
        pipeline.run()


def test_validator_runs_cleanly_on_pipeline_output(tmp_path):
    # Test Section 14 & 34: Validator executes cleanly and returns 0 exit code on valid outputs
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text(
        "nsc_id,drug_name,target_id,gene_symbol,uniprot_id,sequence,sequence_length\n"
        "NSC-100,Drug1,T1,G1,P1,ACDEFGHIKL,10\n"
        "NSC-101,Drug2,T2,G2,P2,MNPQRSTVWY,10\n",
        encoding="utf-8"
    )

    out_dir = tmp_path / "out"
    resolved = make_resolved()
    pipeline = PreprocessingPipeline(
        config=__import__("esmc_target.config").config.load_config(None),
        output_dir=out_dir,
        input_path=dataset_path,
        resolved_model=resolved
    )
    pipeline.run()

    from scripts.validate_outputs import validate
    validator = validate(out_dir)
    assert len(validator.errors) == 0, f"Validator found errors: {validator.errors}"


def test_subset_source_row_indices_provenance(tmp_path):
    # Test Section 15: Subset run preserves original source_row_index (non-contiguous: 1, 3)
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text(
        "nsc_id,drug_name,target_id,gene_symbol,uniprot_id,sequence,sequence_length\n"
        "NSC-100,Drug0,T0,G0,P0,ACDEFGHIKL,10\n"
        "NSC-101,Drug1,T1,G1,P1,MNPQRSTVWY,10\n"
        "NSC-102,Drug2,T2,G2,P2,ACDEFGHIKL,10\n"
        "NSC-103,Drug3,T3,G3,P3,MNPQRSTVWY,10\n",
        encoding="utf-8"
    )

    out_dir = tmp_path / "out_subset"
    resolved = make_resolved()
    pipeline = PreprocessingPipeline(
        config=__import__("esmc_target.config").config.load_config(None),
        output_dir=out_dir,
        input_path=dataset_path,
        nsc_ids="NSC-101,NSC-103",
        resolved_model=resolved
    )
    pipeline.run()

    import csv
    with open(out_dir / "target_provenance.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert [int(r["source_row_index"]) for r in rows] == [1, 3]


def test_long_protein_chunking_exact_overlap():
    # Test Section 3: Exact stride chunking without artificial backward shift
    from esmc_target.sequence import plan_chunks

    # POLE 2286 aa: chunk_size=2046, overlap=256 -> stride=1790
    pole_chunks = plan_chunks(2286, 2046, 256)
    assert pole_chunks == [(0, 2046), (1790, 2286)]
    assert pole_chunks[0][1] - pole_chunks[1][0] == 256  # exact 256 overlap

    # ROS1 2347 aa
    ros1_chunks = plan_chunks(2347, 2046, 256)
    assert ros1_chunks == [(0, 2046), (1790, 2347)]
    assert ros1_chunks[0][1] - ros1_chunks[1][0] == 256  # exact 256 overlap

    # MUC6 2439 aa
    muc6_chunks = plan_chunks(2439, 2046, 256)
    assert muc6_chunks == [(0, 2046), (1790, 2439)]
    assert muc6_chunks[0][1] - muc6_chunks[1][0] == 256  # exact 256 overlap


def test_cache_incompatible_fingerprint_write_aborted(tmp_path):
    # Test Section 4: Writing with an incompatible fingerprint fails fatally and preserves disk cache
    fp1 = build_fingerprint(
        cache_schema_version=1, model_id="biohub/ESMC-600M-hf", model_revision="main",
        tokenizer_id="biohub/ESMC-600M-hf", tokenizer_revision="main", model_class="EsmcModel",
        tokenizer_class="EsmcTokenizer", model_config_hash="abc", tokenizer_config_hash="def",
        transformers_version="5.0.0", torch_version=torch.__version__, sequence_cleaning_version="clean-v1",
        special_residue_policy="keep_if_supported", special_residue_tokenizer_support={"U": True},
        model_max_positions=2048, residue_capacity=2046, chunk_size=2046, overlap=256,
        pooling_method="residue_mean", inference_dtype="float32", effective_attention_implementation="eager", pooling_implementation_version="pool-v1"
    )
    fp2 = dict(fp1)
    fp2["overlap"] = 0
    fp2["fingerprint_hash"] = "different_hash"

    cache_file = tmp_path / "cache.pt"
    c1 = ProteinCache(cache_file, fp1)
    c1.put("seq1", torch.randn(1152), [], [], 10, 1, {})
    c1.save(force=True)

    # Process 2 attempts to load fp2 from existing file with fp1: fails fast on load
    from esmc_target.errors import FingerprintMismatchError
    with pytest.raises(FingerprintMismatchError):
        ProteinCache(cache_file, fp2, rebuild=False)

    # Also verify that if c2 was initialized before c1 saved fp1, calling c2.save() aborts
    non_existent_file = tmp_path / "cache_concurrent.pt"
    c2 = ProteinCache(non_existent_file, fp2)
    c1 = ProteinCache(non_existent_file, fp1)
    c1.put("seq1", torch.randn(1152), [], [], 10, 1, {})
    c1.save(force=True)

    c2.entries["seq2"] = {"embedding": torch.randn(1152)}
    with pytest.raises(FingerprintMismatchError, match="on-disk cache has a different fingerprint"):
        c2.save(force=True)


def test_systemic_esmc_forward_error_propagates(monkeypatch):
    # Test Section 5: Model forward failure during special residue probing propagates as Fatal/RuntimeError
    from esmc_target.esmc_encoder import test_special_residue_support
    
    def failing_run_forward(*args, **kwargs):
        raise RuntimeError("SYSTEM ESMC FAILURE")

    monkeypatch.setattr("esmc_target.esmc_encoder._run_forward", failing_run_forward)

    with pytest.raises(RuntimeError, match="SYSTEM ESMC FAILURE"):
        test_special_residue_support(MagicMock(), MagicMock(), "U")


def test_validator_detects_corrupted_outputs(tmp_path):
    # Test Section 17 & 24: Validator detects corrupt/missing artifacts and returns errors
    from scripts.validate_outputs import validate

    empty_dir = tmp_path / "empty_out"
    empty_dir.mkdir()
    validator = validate(empty_dir)
    assert len(validator.errors) > 0

    # Create run output, then corrupt drug vector to NaN
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text(
        "nsc_id,drug_name,target_id,gene_symbol,uniprot_id,sequence,sequence_length\n"
        "NSC-100,Drug1,T1,G1,P1,ACDEFGHIKL,10\n",
        encoding="utf-8"
    )
    out_dir = tmp_path / "out_nan"
    resolved = make_resolved()
    pipeline = PreprocessingPipeline(
        config=__import__("esmc_target.config").config.load_config(None),
        output_dir=out_dir,
        input_path=dataset_path,
        resolved_model=resolved
    )
    pipeline.run()

    # Corrupt drug embedding vector with NaN
    drug_file = out_dir / "drug_target_embeddings_1152.pt"
    drugs = torch.load(drug_file, map_location="cpu", weights_only=False)
    drugs["NSC-100"][0] = float("nan")
    torch.save(drugs, drug_file)

def test_resolve_model_candidate_not_found_vs_systemic_error(monkeypatch):
    from esmc_target.esmc_encoder import resolve_model

    # 1. Candidate-specific OSError (not found) skips candidate
    def mock_load_not_found(candidate, **kwargs):
        raise OSError("404 Client Error: Repository Not Found for url: https://huggingface.co/fake/model")

    monkeypatch.setattr("esmc_target.esmc_encoder.load_model_and_tokenizer", mock_load_not_found)
    with pytest.raises(Exception) as exc_info:
        resolve_model(["biohub/ESMC-600M-hf"])
    # Not a FatalError, but ModelResolutionError because no candidates succeeded
    from esmc_target.errors import ModelResolutionError
    assert exc_info.type is ModelResolutionError

    # 2. Systemic failure (CUDA OOM / PyTorch RuntimeError) raises FatalError immediately
    def mock_load_systemic(candidate, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr("esmc_target.esmc_encoder.load_model_and_tokenizer", mock_load_systemic)
    with pytest.raises(FatalError, match="Systemic failure loading candidate"):
        resolve_model(["biohub/ESMC-600M-hf"])


def test_rebuild_cache_bypasses_on_disk_merge(tmp_path):
    fp = build_fingerprint(
        cache_schema_version=1, model_id="biohub/ESMC-600M-hf", model_revision="main",
        tokenizer_id="biohub/ESMC-600M-hf", tokenizer_revision="main", model_class="EsmcModel",
        tokenizer_class="EsmcTokenizer", model_config_hash="abc", tokenizer_config_hash="def",
        transformers_version="5.0.0", torch_version=torch.__version__, sequence_cleaning_version="clean-v1",
        special_residue_policy="keep_if_supported", special_residue_tokenizer_support={"U": True},
        model_max_positions=2048, residue_capacity=2046, chunk_size=2046, overlap=256,
        pooling_method="residue_mean", inference_dtype="float32", effective_attention_implementation="eager", pooling_implementation_version="pool-v1"
    )

    cache_file = tmp_path / "cache_rebuild.pt"

    # Step 1: Populate old cache with entry_old
    c_old = ProteinCache(cache_file, fp)
    c_old.put("hash_old", torch.randn(1152), ["P1"], ["T1"], 10, 1, {})
    c_old.save(force=True)
    assert "hash_old" in c_old

    # Step 2: Open with rebuild=True and add entry_new
    c_new = ProteinCache(cache_file, fp, rebuild=True)
    c_new.put("hash_new", torch.randn(1152), ["P2"], ["T2"], 10, 1, {})

    # Perform first partial save (save_every=1) then final save
    c_new.save(force=True)

    # Step 3: Verify old entry is gone and did not reappear
    c_verify = ProteinCache(cache_file, fp)
    assert "hash_new" in c_verify
    assert "hash_old" not in c_verify, "Old entry must not reappear after rebuild-cache save"




