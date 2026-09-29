"""Drug-level MEAN aggregation, within-drug dedup, no-successful-target handling."""

from __future__ import annotations

import math

import pytest
import torch

from esmc_target.aggregation import aggregate_drug_embeddings
from esmc_target.data import RowRecord
from esmc_target.errors import FatalError
from esmc_target.sequence import sequence_hash

DIM = 1152


def row(index: int, nsc_id: str, sequence: str, status: str = "success", **kwargs) -> RowRecord:
    return RowRecord(source_row_index=index, nsc_id=nsc_id, sequence_raw=sequence,
                     sequence=sequence if status != "failed" else None,
                     sequence_hash=sequence_hash(sequence) if status != "failed" else None,
                     actual_length=len(sequence) if status != "failed" else None,
                     status=status, drug_name=kwargs.get("drug_name", "Drug"),
                     target_id=kwargs.get("target_id"), uniprot_id=kwargs.get("uniprot_id"))


def vector(value: float) -> torch.Tensor:
    return torch.full((DIM,), value, dtype=torch.float32)


def test_mean_over_unique_targets():
    rows = [row(0, "NSC-1", "ACDEF"), row(1, "NSC-1", "GHIKL")]
    embeddings = {sequence_hash("ACDEF"): vector(1.0), sequence_hash("GHIKL"): vector(3.0)}
    result = aggregate_drug_embeddings(rows, embeddings, "mean", embedding_dim=DIM)
    assert set(result.drug_embeddings) == {"NSC-1"}
    assert torch.allclose(result.drug_embeddings["NSC-1"], vector(2.0))
    meta = result.drug_metadata["NSC-1"]
    assert meta["num_target_proteins"] == 2
    assert meta["num_unique_target_proteins"] == 2
    assert meta["num_successful_targets"] == 2
    assert meta["num_duplicate_targets_removed"] == 0
    assert meta["aggregation_method"] == "mean"
    assert meta["embedding_dim"] == DIM


def test_within_drug_duplicates_removed_by_sequence_hash():
    same = "ACDEFGHIKL"
    rows = [row(0, "NSC-1", same), row(1, "NSC-1", same), row(2, "NSC-1", "GGGG")]
    embeddings = {sequence_hash(same): vector(2.0), sequence_hash("GGGG"): vector(4.0)}
    result = aggregate_drug_embeddings(rows, embeddings, "mean", embedding_dim=DIM)
    # the duplicated sequence counts once: (2 + 4) / 2, not (2+2+4)/3
    assert torch.allclose(result.drug_embeddings["NSC-1"], vector(3.0))
    assert result.drug_metadata["NSC-1"]["num_duplicate_targets_removed"] == 1
    assert result.within_drug_duplicates_removed == 1


def test_drug_without_successful_targets_is_omitted_not_zero_filled():
    rows = [row(0, "NSC-OK", "ACDEF"), row(1, "NSC-BAD", "XXXX", status="failed")]
    embeddings = {sequence_hash("ACDEF"): vector(1.0)}
    result = aggregate_drug_embeddings(rows, embeddings, "mean", embedding_dim=DIM)
    assert set(result.drug_embeddings) == {"NSC-OK"}
    assert "NSC-BAD" not in result.drug_embeddings
    assert result.drugs_without_successful_targets == ["NSC-BAD"]
    assert result.drug_metadata["NSC-BAD"]["status"] == "failed"
    assert "modality mask" in result.drug_metadata["NSC-BAD"]["error"]
    for vector_ in result.drug_embeddings.values():
        assert float(vector_.norm()) > 0 and torch.isfinite(vector_).all()


def test_aggregation_is_permutation_invariant():
    rows = [row(0, "NSC-1", "ACDEF"), row(1, "NSC-1", "GHIKL"), row(2, "NSC-1", "WWWW")]
    embeddings = {sequence_hash("ACDEF"): vector(1.0), sequence_hash("GHIKL"): vector(2.0),
                  sequence_hash("WWWW"): vector(3.0)}
    first = aggregate_drug_embeddings(rows, embeddings, "mean", embedding_dim=DIM)
    second = aggregate_drug_embeddings(list(reversed(rows)), embeddings, "mean", embedding_dim=DIM)
    assert torch.equal(first.drug_embeddings["NSC-1"], second.drug_embeddings["NSC-1"])


def test_grouping_is_by_nsc_id_not_drug_name():
    rows = [row(0, "NSC-1", "ACDEF", drug_name="Same Name"),
            row(1, "NSC-2", "GHIKL", drug_name="Same Name")]
    embeddings = {sequence_hash("ACDEF"): vector(1.0), sequence_hash("GHIKL"): vector(5.0)}
    result = aggregate_drug_embeddings(rows, embeddings, "mean", embedding_dim=DIM)
    assert len(result.drug_embeddings) == 2, "same drug_name must not merge two nsc_id groups"
    assert torch.allclose(result.drug_embeddings["NSC-1"], vector(1.0))
    assert torch.allclose(result.drug_embeddings["NSC-2"], vector(5.0))


def test_failed_row_counted_but_excluded_from_the_mean():
    rows = [row(0, "NSC-1", "ACDEF"), row(1, "NSC-1", "###", status="failed")]
    embeddings = {sequence_hash("ACDEF"): vector(6.0)}
    result = aggregate_drug_embeddings(rows, embeddings, "mean", embedding_dim=DIM)
    meta = result.drug_metadata["NSC-1"]
    assert meta["num_target_proteins"] == 2
    assert meta["num_successful_targets"] == 1
    assert meta["num_failed_targets"] == 1
    assert torch.allclose(result.drug_embeddings["NSC-1"], vector(6.0))


def test_only_mean_is_accepted():
    rows = [row(0, "NSC-1", "ACDEF")]
    with pytest.raises(FatalError) as excinfo:
        aggregate_drug_embeddings(rows, {}, method="attention", embedding_dim=DIM)
    assert "attention" in str(excinfo.value)


def test_no_attention_aggregator_exists_in_this_repo():
    import esmc_target.aggregation as aggregation
    import esmc_target.projection as projection
    sources = open(aggregation.__file__, encoding="utf-8").read() + \
        open(projection.__file__, encoding="utf-8").read()
    assert "nn.MultiheadAttention" not in sources
    assert "def train_projection" not in sources
