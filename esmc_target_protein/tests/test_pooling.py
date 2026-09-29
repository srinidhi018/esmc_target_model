"""Residue alignment, mean pooling, chunk accumulation, embedding diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from conftest import HIDDEN, MAX_POSITIONS, MockEsmcConfig, MockEsmcModel, MockTokenizer, protein
from esmc_target.errors import AlignmentError, CapacityError, EmbeddingQualityError
from esmc_target.esmc_encoder import (
    TargetProteinEncoder,
    derive_residue_positions,
    encode_chunk,
    extract_residue_hidden_states,
    inspect_runtime,
    probe_special_residue_support,
    residue_field_name,
    residue_token_indices,
    verify_residue_order,
)
from esmc_target.esmc_encoder import test_special_residue_support as check_special_residue_support
from esmc_target.pooling import (
    ResidueAccumulator,
    compute_diagnostics,
    pool_chunks,
    residue_mean_pool,
    validate_embedding,
)
from esmc_target.sequence import derive_residue_capacity, derive_special_token_count, plan_chunks


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

def test_alignment_on_acdefghikl_uses_real_tokenizer_output():
    tokenizer = MockTokenizer()
    model = MockEsmcModel()
    inspection = inspect_runtime(model, tokenizer, "ACDEFGHIKL")
    report = inspection.tokenization
    assert report.tokens[0] == "<cls>" and report.tokens[-1] == "<eos>"
    assert report.num_added_special_tokens == 2
    assert report.residue_tokens == list("ACDEFGHIKL")
    assert report.residue_token_indices == list(range(1, 11))
    assert inspection.hidden_state_shape == [12, HIDDEN]
    assert inspection.special_tokens == 2
    assert inspection.residue_capacity == MAX_POSITIONS - 2 == 2046


def test_special_tokens_are_never_pooled():
    tokenizer, model = MockTokenizer(), MockEsmcModel()
    inspection = inspect_runtime(model, tokenizer, "ACDEFGHIKL")
    ids = torch.tensor([encode_chunk(tokenizer, "ACDEFGHIKL")["input_ids"]])
    out = model(input_ids=ids)
    positions = inspection.tokenization.residue_token_indices
    residues = extract_residue_hidden_states(out, ids, positions, 10)
    assert residues.shape == (10, HIDDEN)
    assert 0 not in positions and 11 not in positions  # CLS / EOS excluded
    all_positions = list(range(12))
    everything = extract_residue_hidden_states(out, ids, all_positions, 12)
    assert not torch.allclose(everything.mean(dim=0), residues.mean(dim=0),
                             atol=1e-4), "pooling over all tokens would differ from residues only"


def test_hidden_state_length_must_match_token_length():
    class BadOutput:
        last_hidden_state = torch.zeros((5, HIDDEN))

    with pytest.raises(AlignmentError) as excinfo:
        extract_residue_hidden_states(BadOutput(), torch.zeros((1, 10), dtype=torch.long),
                                      [1, 2, 3], 3)
    assert "positions" in str(excinfo.value)


def test_residue_positions_come_from_tokenizer_not_a_slice():
    tokenizer = MockTokenizer()
    encoded = encode_chunk(tokenizer, "ACDEF")
    positions = residue_token_indices(encoded["input_ids"], encoded["special_tokens_mask"],
                                      encoded.get("attention_mask"))
    assert positions == [1, 2, 3, 4, 5]
    with pytest.raises(AlignmentError):
        derive_residue_positions(encoded, 4)


def test_residue_order_verification_detects_shift():
    tokenizer = MockTokenizer()
    encoded = encode_chunk(tokenizer, "ACDEF")
    positions = residue_token_indices(encoded["input_ids"], encoded["special_tokens_mask"],
                                      encoded.get("attention_mask"))
    assert verify_residue_order(encoded, positions, "ACDEF") == list("ACDEF")
    with pytest.raises(AlignmentError):
        verify_residue_order(encoded, positions, "ACDFF")


def test_residue_field_is_discovered_at_runtime():
    model, tokenizer = MockEsmcModel(), MockTokenizer()
    out = model(input_ids=torch.tensor([encode_chunk(tokenizer, "ACDEF")["input_ids"]]))
    assert residue_field_name(out) == "last_hidden_state"

    class TupleOutput:
        hidden_states = (torch.zeros((1, 6, HIDDEN)), torch.zeros((1, 6, HIDDEN)))

    assert residue_field_name(TupleOutput()) == "hidden_states[-1]"


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------

def test_residue_mean_pool_matches_manual_mean():
    states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    pooled = residue_mean_pool(states)
    assert torch.allclose(pooled, states.mean(dim=0))
    with pytest.raises(ValueError):
        residue_mean_pool(torch.zeros((2, 2, 3)))
    with pytest.raises(EmbeddingQualityError):
        residue_mean_pool(torch.zeros((0, 3)))


def test_chunk_accumulator_averages_overlapping_positions():
    """overlap=256 on a 2439-length protein: every position covered, overlaps averaged."""
    length, chunk_size, overlap = 2439, 2046, 256
    intervals = plan_chunks(length, chunk_size, overlap)
    acc = ResidueAccumulator(length=length, hidden_size=4)
    for start, end in intervals:
        states = torch.full((end - start, 4), float(start + 1))
        acc.add(start, end, states)
    coverage = acc.assert_full_coverage()
    per_residue = acc.per_residue_embeddings()
    assert per_residue.shape == (length, 4)
    assert coverage["min_coverage_count"] == 1
    assert coverage["num_residues_pooled"] == length
    assert coverage["total_chunk_residues"] == 2046 + (length - 1790) > length  # overlap -> > length
    # Positions in the overlap region are the average of the two chunk values.
    overlap_start = intervals[1][0]
    assert math.isclose(float(per_residue[overlap_start, 0]),
                        (1.0 + (overlap_start + 1)) / 2, rel_tol=1e-6)


def test_overlap_zero_equals_residue_count_weighted_chunk_average():
    length, chunk_size = 500, 256
    intervals = plan_chunks(length, chunk_size, 0)
    chunks = []
    for i, (start, end) in enumerate(intervals):
        states = torch.full((end - start, 3), float(i + 1))
        chunks.append((start, end, states))
    pooled, coverage = pool_chunks(chunks, length, 3)
    assert coverage["total_chunk_residues"] == length
    weighted = sum(float(states.sum()) for _s, _e, states in chunks) / length
    assert math.isclose(float(pooled[0]), weighted / 3.0, rel_tol=1e-5)


def test_accumulator_rejects_uncovered_positions():
    acc = ResidueAccumulator(length=10, hidden_size=2)
    acc.add(0, 5, torch.ones((5, 2)))
    with pytest.raises(EmbeddingQualityError) as excinfo:
        acc.assert_full_coverage()
    assert "never encoded" in str(excinfo.value)


def test_accumulator_rejects_mismatched_chunk_width():
    acc = ResidueAccumulator(length=10, hidden_size=2)
    with pytest.raises(ValueError):
        acc.add(0, 5, torch.ones((4, 2)))


# ---------------------------------------------------------------------------
# numerical sanity + diagnostics
# ---------------------------------------------------------------------------

def test_validate_embedding_rejects_nan_inf_zero_and_wrong_dim():
    good = torch.ones(HIDDEN)
    validate_embedding(good, HIDDEN)
    with pytest.raises(EmbeddingQualityError):
        validate_embedding(torch.full((HIDDEN,), float("nan")), HIDDEN)
    with pytest.raises(EmbeddingQualityError):
        validate_embedding(torch.full((HIDDEN,), float("inf")), HIDDEN)
    with pytest.raises(EmbeddingQualityError):
        validate_embedding(torch.zeros(HIDDEN), HIDDEN)
    with pytest.raises(EmbeddingQualityError):
        validate_embedding(torch.ones(256), HIDDEN)
    with pytest.raises(EmbeddingQualityError):
        validate_embedding(torch.ones(4, HIDDEN), HIDDEN)


def test_embedding_diagnostics_on_mocked_embeddings():
    """Section 9/21A diagnostics on mocked embeddings (mean, std, L2 norm)."""
    vector = torch.arange(HIDDEN, dtype=torch.float32) / 1000.0 + 1.0
    diagnostics = compute_diagnostics(vector)
    expected_mean = float(vector.mean())
    expected_norm = float(vector.norm())
    assert math.isclose(diagnostics.embedding_mean, expected_mean, rel_tol=1e-6)
    assert math.isclose(diagnostics.embedding_l2_norm, expected_norm, rel_tol=1e-6)
    assert math.isclose(diagnostics.embedding_std,
                        float(vector.std(unbiased=False)), rel_tol=1e-6)
    assert diagnostics.embedding_l2_norm > 0
    assert set(diagnostics.to_dict()) == {"embedding_mean", "embedding_std", "embedding_l2_norm"}


def test_hidden_size_1152_is_asserted_and_shapes_derive_from_config():
    from conftest import make_resolved
    from esmc_target.errors import ModelResolutionError

    # 1152 is an acceptance invariant of ESMC-600M, asserted once.
    with pytest.raises(ModelResolutionError):
        TargetProteinEncoder(resolved=make_resolved(hidden_size=64), device=torch.device("cpu"),
                             chunk_size="auto", overlap=8)

    # Everything else derives from model.config: positions, capacity, tensors.
    resolved = make_resolved(hidden_size=1152, max_position_embeddings=128)
    enc = TargetProteinEncoder(resolved=resolved, device=torch.device("cpu"),
                               chunk_size="auto", overlap=8)
    assert enc.hidden_size == 1152
    assert enc.model_max_positions == 128
    assert enc.residue_capacity == 126          # 128 - 2 measured special tokens
    assert enc.chunk_size == 126
    out = enc.encode("ACDEFGHIKL")
    assert out.embedding.shape == (1152,) and out.embedding.dtype == torch.float32

    # a manually lowered chunk size is honoured, a higher one is rejected
    small = TargetProteinEncoder(resolved=resolved, device=torch.device("cpu"),
                                 chunk_size=32, overlap=8)
    assert small.residue_capacity == 126
    assert small.encode(protein(100)).num_chunks == 4
    assert small.encode("ACDEFGHIKL").num_chunks == 1
    with pytest.raises(CapacityError):
        TargetProteinEncoder(resolved=resolved, device=torch.device("cpu"),
                             chunk_size=200, overlap=8)


# ---------------------------------------------------------------------------
# special-residue support through the complete path
# ---------------------------------------------------------------------------

def test_special_residue_alignment_keeps_neighbours_unshifted():
    """A supported symbol must not shift, merge or drop any neighbour residue."""
    tokenizer, model = MockTokenizer(), MockEsmcModel()
    for symbol in "XBZJUO":
        result = check_special_residue_support(model, tokenizer, symbol)
        assert result.primary_criterion_passed, f"{symbol}: {result.detail}"
        assert result.residue_tokens == list(f"ACD{symbol}EFG"), result.residue_tokens
        assert len(result.residue_tokens) == 7
        assert result.supported is True


def test_special_residue_unsupported_when_mapped_to_unk_id():
    tokenizer, model = MockTokenizer(unk_tokens=("U",)), MockEsmcModel()
    result = check_special_residue_support(model, tokenizer, "U")
    # The position is still occupied and the neighbours are unshifted...
    assert result.primary_criterion_passed is True
    assert result.residue_tokens == list("ACDUEFG")
    # ...but the token id is the unk id, so the symbol is NOT supported.
    assert result.diagnostic_unk is False
    assert result.supported is False
    assert "unk" in result.detail.lower()
    support = probe_special_residue_support(model, tokenizer)
    assert support["U"].supported is False
    assert support["O"].supported is True


def test_special_residue_support_map_only_lists_probed_symbols():
    tokenizer, model = MockTokenizer(unk_tokens=("O",)), MockEsmcModel()
    support = {s: r.supported for s, r in probe_special_residue_support(model, tokenizer).items()}
    assert set(support) == set("XBZJUO")
    assert support["O"] is False and support["U"] is True


# ---------------------------------------------------------------------------
# encoder integration with the mock
# ---------------------------------------------------------------------------

def test_encoder_short_sequence_single_chunk(encoder):
    out = encoder.encode("ACDEFGHIKL")
    assert out.num_chunks == 1
    assert out.chunk_boundaries == [(0, 10)]
    assert out.embedding.shape == (HIDDEN,)
    assert out.embedding.dtype == torch.float32
    assert out.coverage["num_residues_pooled"] == 10


def test_encoder_chunked_protein_full_coverage(encoder):
    sequence = protein(2439)
    out = encoder.encode(sequence)
    assert out.num_chunks == 2
    assert out.coverage["num_residues_pooled"] == 2439
    assert out.coverage["min_coverage_count"] >= 1
    assert out.coverage["total_chunk_residues"] == 2046 + (2439 - 1790)
    assert out.embedding.shape == (HIDDEN,)


def test_encoder_never_sends_more_positions_than_the_model_allows(encoder):
    encoder.encode(protein(2439))
    assert encoder.model.max_positions_seen <= MAX_POSITIONS


def test_encoder_is_frozen_and_refuses_train_mode(encoder):
    assert all(not p.requires_grad for p in encoder.model.parameters())
    assert encoder.model.training is False
    with pytest.raises(Exception):
        encoder.train(True)


def test_encoder_forward_pass_count_per_chunk(encoder):
    before = encoder.chunk_forward_passes
    encoder.encode("ACDEFGHIKL")
    encoder.encode(protein(2439))
    assert encoder.chunk_forward_passes - before == 3  # 1 + 2 chunks
