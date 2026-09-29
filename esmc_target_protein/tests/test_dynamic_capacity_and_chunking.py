"""Unit tests for dynamic capacity, chunking, overlap reconstruction, and SHA256 identity (Tests V - AA)."""

import pytest
import torch
from esmc_target.errors import CapacityError
from esmc_target.pooling import ResidueAccumulator, pool_chunks
from esmc_target.sequence import (
    derive_residue_capacity,
    plan_chunks,
    resolve_chunk_size,
    sequence_hash,
)


def test_dynamic_residue_capacity():
    # Test V: residue_capacity = model_max_positions - special_token_count
    cap = derive_residue_capacity(model_max_positions=2048, special_tokens=2)
    assert cap == 2046

    with pytest.raises(CapacityError, match="special token count must be >= 0"):
        derive_residue_capacity(2048, -1)

    with pytest.raises(CapacityError, match="Derived residue capacity is"):
        derive_residue_capacity(2, 2)


def test_invalid_max_position_override_fails():
    # Test W: chunk_size override > residue_capacity fails
    residue_cap = 1000
    with pytest.raises(CapacityError, match="exceeds the derived residue capacity"):
        resolve_chunk_size(configured=1024, residue_capacity=residue_cap)

    # Valid chunk_size override succeeds
    assert resolve_chunk_size(configured=500, residue_capacity=residue_cap) == 500
    assert resolve_chunk_size(configured="auto", residue_capacity=residue_cap) == 1000


def test_long_sequence_chunking_and_overlap_reconstruction():
    # Test X & Y: plan chunks and accumulate with overlap
    L = 2500
    chunk_size = 1000
    overlap = 200
    intervals = plan_chunks(length=L, chunk_size=chunk_size, overlap=overlap)
    
    assert intervals[0][0] == 0
    assert intervals[-1][1] == L
    
    # Simulate constant residue vectors (e.g., all 1.0s)
    hidden_size = 1152
    chunk_tuples = []
    for start, end in intervals:
        chunk_len = end - start
        states = torch.ones((chunk_len, hidden_size), dtype=torch.float32)
        chunk_tuples.append((start, end, states))

    pooled, coverage = pool_chunks(chunk_tuples, length=L, hidden_size=hidden_size)
    
    assert coverage["num_residues_pooled"] == L
    assert coverage["min_coverage_count"] >= 1
    # Because all residue vectors are 1.0s, the mean over overlapping chunks divided by count is still exactly 1.0s
    assert torch.allclose(pooled, torch.ones(hidden_size))


def test_no_duplicate_weighting_from_overlap():
    # Test Z: position-wise overlap division ensures no double-counting
    acc = ResidueAccumulator(length=10, hidden_size=4)
    
    # Chunk 1: [0, 6) with value 2.0
    c1 = torch.full((6, 4), 2.0)
    acc.add(0, 6, c1)
    
    # Chunk 2: [4, 10) with value 4.0
    c2 = torch.full((6, 4), 4.0)
    acc.add(4, 10, c2)

    per_residue = acc.per_residue_embeddings()
    # Positions 0-3 covered 1 time with 2.0 -> 2.0
    assert torch.allclose(per_residue[0:4], torch.full((4, 4), 2.0))
    # Positions 4-5 covered 2 times with (2.0 + 4.0)/2 = 3.0
    assert torch.allclose(per_residue[4:6], torch.full((2, 4), 3.0))
    # Positions 6-9 covered 1 time with 4.0 -> 4.0
    assert torch.allclose(per_residue[6:10], torch.full((4, 4), 4.0))


def test_protein_sha256_identity():
    # Test AA: SHA256 of cleaned sequence
    seq1 = "ACDEFGHIKLMNPQRSTVWY"
    seq2 = "ACDEFGHIKLMNPQRSTVWY"
    seq3 = "ACDEFGHIKLMNPQRSTVW"
    
    hash1 = sequence_hash(seq1)
    hash2 = sequence_hash(seq2)
    hash3 = sequence_hash(seq3)
    
    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64
