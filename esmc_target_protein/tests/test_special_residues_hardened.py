"""Unit tests for special residue support and X handling (Tests H - P)."""

import pytest
from esmc_target.errors import UnsupportedResidueError
from esmc_target.sequence import clean_sequence


def test_special_residues_support_natively_supported():
    # Test support for X, B, Z, J, U, O when tokenizer supports them
    support_map = {"X": True, "B": True, "Z": True, "J": True, "U": True, "O": True}
    res = clean_sequence("ACDXBZJUOEFG", policy="keep_if_supported", tokenizer_support=support_map)
    assert res.cleaned == "ACDXBZJUOEFG"
    assert len(res.events) == 6
    for event in res.events:
        assert event.tokenizer_encoded_natively is True
        assert event.replaced_with is None


def test_unsupported_x_with_keep_if_supported_fails():
    # Test N: unsupported X with keep_if_supported fails clearly
    support_map = {"X": False}
    with pytest.raises(UnsupportedResidueError, match="special residue 'X' at position 3 is not supported"):
        clean_sequence("ACDXEFG", policy="keep_if_supported", tokenizer_support=support_map)


def test_unsupported_b_z_j_u_o_with_replace_with_x_works():
    # Test O: unsupported B/Z/J/U/O with replace_with_X works and records substitution
    support_map = {"B": False, "Z": False, "J": False, "U": False, "O": False, "X": True}
    res = clean_sequence("ACDBZJUOEFG", policy="replace_with_X", tokenizer_support=support_map)
    assert res.cleaned == "ACDXXXXXEFG"
    assert len(res.events) == 5
    for event in res.events:
        assert event.tokenizer_encoded_natively is False
        assert event.replaced_with == "X"
        assert isinstance(event.tokenizer_encoded_natively, bool)


def test_special_residue_provenance_boolean_field():
    # Test P: special-residue provenance contains true/false, never unexplained null/None
    support_map = {"U": True, "B": False, "X": True}
    res = clean_sequence("ACDUBEFG", policy="replace_with_X", tokenizer_support=support_map)
    payload = res.events_payload()
    assert len(payload) == 2
    
    # U event (natively supported)
    assert payload[0]["symbol"] == "U"
    assert payload[0]["tokenizer_encoded_natively"] is True
    assert payload[0]["replaced_with"] is None
    
    # B event (replaced with X)
    assert payload[1]["symbol"] == "B"
    assert payload[1]["tokenizer_encoded_natively"] is False
    assert payload[1]["replaced_with"] == "X"
