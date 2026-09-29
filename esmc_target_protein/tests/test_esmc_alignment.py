"""Tests for ESMC special token handling, tokenizer alignment, and hidden size derivation."""

from __future__ import annotations

import pytest
import torch

from esmc_target.errors import AlignmentError, CapacityError, ModelResolutionError
from esmc_target.esmc_encoder import (
    TargetProteinEncoder,
    derive_residue_positions,
    encode_chunk,
    extract_residue_hidden_states,
    residue_token_indices,
    verify_residue_order,
)
from conftest import MockTokenizer, make_resolved


def test_A_tokenizer_requests_special_tokens_mask():
    tok = MockTokenizer()
    encoded = encode_chunk(tok, "ACDEF")
    assert "special_tokens_mask" in encoded
    assert "attention_mask" in encoded
    assert encoded["special_tokens_mask"] == [1, 0, 0, 0, 0, 0, 1]


def test_B_missing_special_tokens_mask_fails_safely():
    encoded = {"input_ids": [0, 1, 2, 3], "attention_mask": [1, 1, 1, 1]}
    with pytest.raises(AlignmentError) as exc:
        derive_residue_positions(encoded, 2)
    assert "special_tokens_mask is missing" in str(exc.value)


def test_C_attention_mask_not_accepted_as_substitute_for_special_tokens_mask():
    ids = [0, 10, 11, 1]
    attn = [1, 1, 1, 1]
    with pytest.raises(AlignmentError) as exc:
        residue_token_indices(ids, None, attn)
    assert "special_tokens_mask is required" in str(exc.value)


def test_D_residue_alignment_exact():
    tok = MockTokenizer()
    seq = "ACDEFG"
    encoded = encode_chunk(tok, seq)
    positions = derive_residue_positions(encoded, len(seq))
    assert len(positions) == len(seq)
    assert positions == [1, 2, 3, 4, 5, 6]


def test_E_residue_order_exact():
    tok = MockTokenizer()
    seq = "ACDEFG"
    encoded = encode_chunk(tok, seq)
    positions = derive_residue_positions(encoded, len(seq))
    order = verify_residue_order(encoded, positions, seq)
    assert order == list("ACDEFG")


def test_F_hidden_size_derived_correctly():
    resolved = make_resolved(hidden_size=1152)
    encoder = TargetProteinEncoder(resolved, device=torch.device("cpu"))
    assert encoder.hidden_size == 1152


def test_G_hidden_size_mismatch_fails():
    resolved = make_resolved(hidden_size=768)
    with pytest.raises(ModelResolutionError):
        TargetProteinEncoder(resolved, device=torch.device("cpu"))
