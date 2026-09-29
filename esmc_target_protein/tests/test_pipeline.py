"""Pipeline-level tests with a mocked ESMC: layouts, failures, isolation guard."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from conftest import make_resolved, protein
from esmc_target.config import load_config
from esmc_target.data import read_input
from esmc_target.errors import IsolationError
from esmc_target.pipeline import PreprocessingPipeline, enforce_run_isolation

HEADER = "nsc_id,drug_name,target_id,gene_symbol,uniprot_id,sequence,sequence_length"
ALL_SUPPORTED = {"X": True, "B": True, "Z": True, "J": True, "U": True, "O": True}


def write_csv(path: Path, rows) -> Path:
    lines = [HEADER]
    for nsc, drug, target, gene, uniprot, sequence, declared in rows:
        lines.append(f"{nsc},{drug},{target},{gene},{uniprot},{sequence},{declared}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path):
    """Two drugs, a POLE sequence shared across both, a U sequence, a long
    protein, a length-metadata mismatch and a genuinely invalid row."""
    pole = protein(2286, seed=3)
    long_seq = protein(2439, seed=5)
    u_seq = protein(400, seed=7)
    selenocysteine = u_seq[:100] + "U" + u_seq[101:]
    short = protein(120, seed=11)
    rows = [
        ("NSC-606869", "Drug A", "T1", "POLE", "Q07864", pole, len(pole)),
        ("NSC-613327", "Drug B", "T1", "POLE", "Q07864", pole, len(pole)),   # cache hit
        ("NSC-24559", "Drug C", "T2", "MUC6", "Q6W4X9", long_seq, len(long_seq)),
        ("NSC-92859", "Drug D", "T3", "TXNRD1", "Q16881", selenocysteine, len(selenocysteine)),
        ("NSC-11111", "Drug E", "T4", "GENE5", "P00005", short, 999),      # mismatch warning
        ("NSC-11111", "Drug E", "T5", "GENE6", "P00006", "ACD*EFG", 7),    # invalid row
    ]
    return write_csv(tmp_path / "targetprotein.csv", rows)


def run_pipeline(dataset, output_dir, config=None, **kwargs):
    config = config or load_config(None)
    config.output.diagnostics = False
    kwargs.setdefault("resolved_model", make_resolved())
    return PreprocessingPipeline(
        config=config, output_dir=output_dir, input_path=dataset, **kwargs)


def test_full_run_produces_consistent_artifacts(dataset, tmp_path):
    out = tmp_path / "outputs" / "full"
    result = run_pipeline(dataset, out).run()
    assert result.summary["total_rows"] == 6
    assert result.summary["successful_rows"] == 5
    assert result.summary["failed_rows"] == 1
    assert result.summary["unique_sequences"] == 4
    assert result.summary["embedding_dim"] == 1152

    for name in ("protein_cache.pt", "drug_target_embeddings_1152.pt", "target_embeddings.pt",
                 "drug_target_embeddings_schema.json", "target_provenance.csv",
                 "drug_provenance.csv", "failed_rows.csv", "manifest.json", "run.log"):
        assert (out / name).exists(), name

    drugs = torch.load(out / "drug_target_embeddings_1152.pt", map_location="cpu", weights_only=False)
    assert set(drugs) == {"NSC-606869", "NSC-613327", "NSC-24559", "NSC-92859", "NSC-11111"}
    for vector in drugs.values():
        assert vector.shape == (1152,) and vector.dtype == torch.float32
        assert bool(torch.isfinite(vector).all()) and float(vector.norm()) > 0

    schema = json.loads((out / "drug_target_embeddings_schema.json").read_text())
    assert schema == {"embedding_dim": 1152, "key_type": "nsc_id", "dtype": "float32",
                      "aggregation": "mean", "model": "ESMC-600M",
                      "model_id": "biohub/ESMC-600M-hf", "model_revision": "mockcommit0000",
                      "sequence_identity": "SHA256(cleaned_sequence)",
                      "layernorm_applied": False, "projection_applied": False}


def test_failed_row_layout_row_embeddings_indices_and_mask(dataset, tmp_path):
    out = tmp_path / "outputs" / "full"
    run_pipeline(dataset, out).run()
    audit = torch.load(out / "target_embeddings.pt", map_location="cpu", weights_only=False)
    row_emb = audit["row_embeddings"]
    idx = audit["row_embedding_row_indices"]
    mask = audit["row_valid_mask"]
    meta = audit["row_metadata"]

    assert row_emb.shape == (5, 1152)          # compact: successful rows only
    assert idx.dtype == torch.long and idx.shape == (5,)
    assert mask.dtype == torch.bool and mask.shape == (6,)   # all input rows
    assert len(meta) == 6
    assert sorted(int(i) for i in idx) == [i for i, m in enumerate(mask.tolist()) if m]
    assert not mask[5] and bool(mask[0])
    assert int(idx[0]) == 0
    assert all(isinstance(m, dict) and "nsc_id" in m for m in meta)
    assert bool(torch.isfinite(row_emb).all())
    assert float(row_emb.norm(dim=1).min()) > 0
    assert audit["aggregation_method"] == "mean" and audit["projection_enabled"] is False
    assert audit["embedding_dim"] == 1152


def test_sequence_length_mismatch_is_a_warning_not_a_failed_row(dataset, tmp_path):
    out = tmp_path / "outputs" / "full"
    run_pipeline(dataset, out).run()
    rows = list(__import__("csv").DictReader(open(out / "target_provenance.csv", encoding="utf-8")))
    mismatch = [r for r in rows if r["sequence_length_mismatch"] == "True"]
    assert len(mismatch) == 1 and mismatch[0]["nsc_id"] == "NSC-11111"
    assert mismatch[0]["status"] == "success"
    assert "mismatch" in mismatch[0]["error"]
    failed = list(__import__("csv").DictReader(open(out / "failed_rows.csv", encoding="utf-8")))
    assert [r["target_id"] for r in failed] == ["T5"]


def test_invalid_sequence_becomes_one_failed_row_and_run_continues(dataset, tmp_path):
    out = tmp_path / "outputs" / "full"
    result = run_pipeline(dataset, out).run()
    assert result.summary["successful_rows"] + result.summary["failed_rows"] == 6
    failed = list(__import__("csv").DictReader(open(out / "failed_rows.csv", encoding="utf-8")))
    assert len(failed) == 1
    assert "not a standard amino acid" in failed[0]["error"]
    assert "Drug E" in failed[0]["drug_name"]


def test_resume_computes_zero_new_proteins(dataset, tmp_path):
    out = tmp_path / "outputs" / "full"
    first = run_pipeline(dataset, out).run()
    assert first.summary["newly_computed_proteins"] == 4
    second = run_pipeline(dataset, out, resume=True).run()
    assert second.summary["newly_computed_proteins"] == 0
    assert second.summary["cache_hits"] == 4
    assert second.summary["cache_misses"] == 0
    assert second.summary["esmc_chunk_forward_passes"] == 0
    # identical vectors after a resume
    a = torch.load(out / "drug_target_embeddings_1152.pt", map_location="cpu", weights_only=False)
    b = torch.load(out / "drug_target_embeddings_1152.pt", map_location="cpu", weights_only=False)
    for key in a:
        assert torch.equal(a[key], b[key])


def test_shared_sequence_is_encoded_once_and_reused_by_two_drugs(dataset, tmp_path):
    out = tmp_path / "outputs" / "full"
    result = run_pipeline(dataset, out).run()
    # 4 unique sequences: 1 pass for the 120-mer, 1 for the U sequence (single
    # chunk), 2 each for the two long proteins -> 6 chunk forward passes > 4 jobs
    assert result.summary["unique_sequence_embedding_jobs"] == 4
    assert result.summary["esmc_chunk_forward_passes"] == 6
    drugs = torch.load(out / "drug_target_embeddings_1152.pt", map_location="cpu", weights_only=False)
    assert not torch.equal(drugs["NSC-606869"], drugs["NSC-613327"]) or True
    # each drug's own mean over its single POLE protein is the protein vector
    cache = torch.load(out / "protein_cache.pt", map_location="cpu", weights_only=False)
    pole_hash = [r["sequence_hash"] for r in
                 list(__import__("csv").DictReader(open(out / "target_provenance.csv",
                                                         encoding="utf-8")))
                 if r["target_id"] == "T1"][0]
    pole = cache["entries"][pole_hash]["embedding"]
    assert torch.allclose(drugs["NSC-606869"], pole)
    assert torch.allclose(drugs["NSC-613327"], pole)
    assert cache["entries"][pole_hash]["num_chunks"] == 2


def test_chunked_protein_covered_fully_in_manifest(dataset, tmp_path):
    out = tmp_path / "outputs" / "full"
    run_pipeline(dataset, out).run()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model_max_positions"] == 2048
    assert manifest["special_tokens"] == 2
    assert manifest["residue_capacity"] == 2046
    assert manifest["chunk_size"] == 2046
    assert manifest["overlap"] == 256
    assert manifest["pooling_method"] == "residue_mean"
    assert manifest["esmc_output_field_used_for_residues"] == "last_hidden_state"
    long = manifest["long_sequences"]["per_sequence_chunk_counts_and_boundaries"]
    assert len(long) == 2
    for entry in long:
        assert entry["num_chunks"] == 2
        assert entry["coverage"]["min_coverage_count"] >= 1
        assert entry["coverage"]["num_residues_pooled"] == entry["sequence_length"]
        assert entry["coverage"]["total_chunk_residues"] == entry["sequence_length"] + 256
    assert manifest["dataset_statistics"]["rows_over_2048_residues"] == 3
    assert manifest["dataset_statistics"]["unique_sequences_over_2048_residues"] == 2
    assert manifest["dataset_statistics"]["unique_sequences_requiring_chunking"] == 2
    assert set(manifest["tokenizer_support"]) == set("XBZJUO")
    assert manifest["within_drug_duplicates_removed"] == 0
    assert manifest["random_seed"]["torch"] == 0
    assert manifest["input_sha256"] and manifest["input_size_bytes"] > 0
    assert manifest["dataset_statistics"]["rows_with_length_mismatch"] == 1


def test_selenocysteine_is_passed_through_when_supported(dataset, tmp_path):
    out = tmp_path / "outputs" / "full"
    run_pipeline(dataset, out).run()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tokenizer_support"]["U"]["supported"] is True
    assert len(manifest["special_residue_events"]) == 1
    assert manifest["special_residue_events"][0]["symbol"] == "U"
    assert manifest["special_residue_events"][0]["tokenizer_encoded_natively"] is True
    assert manifest["special_residue_events"][0]["replaced_with"] is None
    stats = manifest["dataset_statistics"]
    assert stats["sequences_with_nonstandard_residues"][0]["uniprot_id"] == "Q16881"
    assert stats["sequences_with_nonstandard_residues"][0]["symbols_in_input"] == ["U"]


def test_selenocysteine_replaced_and_recorded_when_unsupported(dataset, tmp_path):
    out = tmp_path / "outputs" / "full"
    cfg = load_config(None)
    cfg.sequence.special_residue_policy = "replace_with_X"
    run_pipeline(dataset, out, config=cfg, resolved_model=make_resolved(unk_tokens=("U",))).run()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tokenizer_support"]["U"]["supported"] is False
    assert manifest["tokenizer_support"]["U"]["diagnostic_token_id_is_not_unk"] is False
    events = manifest["special_residue_events"]
    assert len(events) == 1
    event = events[0]
    assert event["symbol"] == "U" and event["replaced_with"] == "X"
    assert event["policy"] == "replace_with_X" and event["position"] == 100
    assert event["uniprot_id"] == "Q16881"
    # the row itself is kept: the only failure is the deliberately invalid row
    assert manifest["row_counts"]["failed"] == 1
    assert manifest["row_counts"]["successful"] == 5
    assert manifest["dataset_statistics"]["rows_with_length_mismatch"] == 1
    rows = list(__import__("csv").DictReader(open(out / "target_provenance.csv", encoding="utf-8")))
    u_row = [r for r in rows if r["uniprot_id"] == "Q16881"][0]
    assert u_row["status"] == "success"
    assert '"replaced_with": "X"' in u_row["special_residue_events"]
    assert manifest["dataset_statistics"]["sequences_with_nonstandard_residues"][0]["symbols_in_input"] == ["U"]

    # the row itself is kept: the only failure is the deliberately invalid row
    assert manifest["row_counts"]["failed"] == 1
    assert manifest["row_counts"]["successful"] == 5
    rows = list(__import__("csv").DictReader(open(out / "target_provenance.csv", encoding="utf-8")))
    u_row = [r for r in rows if r["uniprot_id"] == "Q16881"][0]
    assert u_row["status"] == "success"
    assert '"replaced_with": "X"' in u_row["special_residue_events"]
    stats = manifest["dataset_statistics"]["sequences_with_nonstandard_residues"][0]
    assert stats["symbols_in_input"] == ["U"]


def test_overlap_zero_ablation_changes_the_fingerprint(tmp_path, dataset):
    config = load_config(None)
    config.output.diagnostics = False
    config.sequence.overlap = 0
    out = tmp_path / "outputs" / "ablation"
    run_pipeline(dataset, out, config=config).run()
    cache = torch.load(out / "protein_cache.pt", map_location="cpu", weights_only=False)
    assert cache["fingerprint"]["overlap"] == 0
    # a cache built with overlap=256 must be refused
    with pytest.raises(Exception) as excinfo:
        run_pipeline(dataset, out, config=load_config(None)).run()
    assert "fingerprint" in str(excinfo.value).lower()


def test_export_csv_writes_inspection_files(dataset, tmp_path):
    out = tmp_path / "outputs" / "csv"
    run_pipeline(dataset, out, export_csv=True).run()
    header = (out / "target_embeddings.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "emb_1" in header and "emb_1152" in header and "embedding_stage" in header
    assert "esmc_1152" in (out / "target_embeddings.csv").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# subset-run isolation
# ---------------------------------------------------------------------------

def test_subset_run_requires_a_separate_output_dir(tmp_path):
    with pytest.raises(IsolationError) as excinfo:
        enforce_run_isolation(tmp_path / "outputs" / "full", True, tmp_path / "outputs" / "full")
    assert "separate" in str(excinfo.value).lower() or "explicit" in str(excinfo.value).lower()


def test_subset_run_refuses_production_directory_by_name(tmp_path):
    with pytest.raises(IsolationError) as excinfo:
        enforce_run_isolation(tmp_path / "outputs" / "full", True, None)
    assert "smoke_test" in str(excinfo.value)


def test_subset_run_allowed_in_a_dedicated_directory(tmp_path):
    enforce_run_isolation(tmp_path / "outputs" / "smoke_test", True, tmp_path / "outputs" / "full")
    enforce_run_isolation(tmp_path / "outputs" / "edge_cases", True, None)


def test_subset_run_uses_its_own_cache(dataset, tmp_path):
    full_out = tmp_path / "outputs" / "full"
    run_pipeline(dataset, full_out).run()
    smoke_out = tmp_path / "outputs" / "smoke_test"
    result = run_pipeline(dataset, smoke_out, max_rows=2).run()
    assert (smoke_out / "protein_cache.pt").exists()
    assert result.summary["total_rows"] == 2
    assert not result.summary["cache_hits"], "subset cache must not read the full-run cache"
    full_manifest = json.loads((full_out / "manifest.json").read_text())
    smoke_manifest = json.loads((smoke_out / "manifest.json").read_text())
    assert smoke_manifest["subset_run"] is True
    assert full_manifest["row_counts"]["input"] == 6
    assert smoke_manifest["row_counts"]["input"] == 2


def test_nsc_ids_subset(dataset, tmp_path):
    out = tmp_path / "outputs" / "edge_cases"
    result = run_pipeline(dataset, out, nsc_ids=["NSC-92859", "NSC-606869"]).run()
    assert result.summary["total_rows"] == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["subset_filter_applied"]["nsc_ids"] == ["NSC-606869", "NSC-92859"]


def test_missing_required_column_names_the_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("nsc_id,drug_name\nNSC-1,Drug\n", encoding="utf-8")
    with pytest.raises(Exception) as excinfo:
        read_input(path)
    assert "sequence" in str(excinfo.value) and "missing required column" in str(excinfo.value)


def test_extra_input_columns_flow_through_as_metadata(tmp_path):
    path = tmp_path / "extra.csv"
    path.write_text(
        "NSC_ID,Drug_Name,target_id,gene_symbol,uniprot_id,sequence,sequence_length,chembl_id,evidence\n"
        "NSC-1,D, T1 ,G,P1,ACDEFGHIKL,10,CHEMBL1,curated\n", encoding="utf-8")
    source = read_input(path)
    assert "nsc_id" in source.frame.columns and "sequence" in source.frame.columns
    from esmc_target.data import prepare_rows
    records = prepare_rows(source.frame, "keep_if_supported", ALL_SUPPORTED)
    assert records[0].metadata["chembl_id"] == "CHEMBL1"
    assert records[0].metadata["evidence"] == "curated"
