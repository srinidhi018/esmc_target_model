"""Unit tests for drug aggregation, deduplication, and missing target handling (Tests AB - AE)."""

import pytest
import torch
from esmc_target.aggregation import aggregate_drug_embeddings
from esmc_target.data import RowRecord


def test_drug_sequence_deduplication_and_mean_aggregation():
    # Test AC & AD: Deduplication within drug and MEAN aggregation
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([3.0, 4.0, 5.0])
    embeddings_by_hash = {
        "hash_seq1": v1,
        "hash_seq2": v2,
    }

    row1 = RowRecord(source_row_index=0, nsc_id="NSC-740", sequence_raw="A", sequence="A",
                     sequence_hash="hash_seq1", status="success")
    row2 = RowRecord(source_row_index=1, nsc_id="NSC-740", sequence_raw="A", sequence="A",
                     sequence_hash="hash_seq1", status="success")  # Duplicate target protein!
    row3 = RowRecord(source_row_index=2, nsc_id="NSC-740", sequence_raw="B", sequence="B",
                     sequence_hash="hash_seq2", status="success")

    result = aggregate_drug_embeddings([row1, row2, row3], embeddings_by_hash)

    assert "NSC-740" in result.drug_embeddings
    assert result.within_drug_duplicates_removed == 1
    
    # Aggregated embedding should be mean of v1 and v2, i.e., ([1,2,3] + [3,4,5])/2 = [2, 3, 4]
    expected_mean = torch.tensor([2.0, 3.0, 4.0])
    assert torch.allclose(result.drug_embeddings["NSC-740"], expected_mean)


def test_no_successful_targets_omitted_from_pt_dict():
    # Test AE: A drug with zero successful targets must NOT receive zero/NaN/random vector, must be omitted from dict
    row_failed = RowRecord(source_row_index=0, nsc_id="NSC-999", sequence_raw="INVALID", status="failed")

    result = aggregate_drug_embeddings([row_failed], embeddings_by_hash={})

    assert "NSC-999" not in result.drug_embeddings
    assert "NSC-999" in result.drugs_without_successful_targets
    assert result.drug_metadata["NSC-999"]["status"] == "failed"


def test_duplicate_sequence_reused_across_uniprot_ids():
    # Test AB: Identical protein sequences with different UniProt/Target IDs map to same sequence_hash
    row_a = RowRecord(source_row_index=0, nsc_id="NSC-1", sequence_raw="SAME_SEQ", sequence="SAME_SEQ",
                      sequence_hash="hash_same", uniprot_id="P12345", target_id="T1", status="success")
    row_b = RowRecord(source_row_index=1, nsc_id="NSC-2", sequence_raw="SAME_SEQ", sequence="SAME_SEQ",
                      sequence_hash="hash_same", uniprot_id="P67890", target_id="T2", status="success")

    v_same = torch.tensor([1.0, 1.0, 1.0])
    embeddings_by_hash = {"hash_same": v_same}

    result = aggregate_drug_embeddings([row_a, row_b], embeddings_by_hash)
    
    assert "NSC-1" in result.drug_embeddings
    assert "NSC-2" in result.drug_embeddings
    assert torch.allclose(result.drug_embeddings["NSC-1"], v_same)
    assert torch.allclose(result.drug_embeddings["NSC-2"], v_same)
