"""Sequence cleaning, hashing, length policy, special residues, chunk planner."""

from __future__ import annotations

import pytest

from conftest import protein
from esmc_target.errors import CapacityError, InvalidSequenceError, UnsupportedResidueError
from esmc_target.sequence import (
    clean_sequence,
    derive_residue_capacity,
    plan_chunks,
    resolve_chunk_size,
    resolve_model_max_positions,
    sequence_hash,
    special_residue_occurrences,
    validate_chunk_plan,
    validate_declared_length,
)

# Capacity derived the way the pipeline derives it: 2048 positions minus the
# 2 special tokens a CLS/EOS tokenizer actually adds.
CAPACITY = derive_residue_capacity(2048, 2)
assert CAPACITY == 2046


# ---------------------------------------------------------------------------
# cleaning + hashing
# ---------------------------------------------------------------------------

def test_clean_basic_and_uppercase():
    assert clean_sequence("acdefghikl").cleaned == "ACDEFGHIKL"
    assert clean_sequence("ACDEFGHIKL").cleaned == "ACDEFGHIKL"


def test_whitespace_variants_hash_and_are_equivalent():
    plain = clean_sequence("ACDEFGHIKL")
    trailing = clean_sequence("ACDEFGHIKL ")
    spaced = clean_sequence(" A C D E F G H I K L ")
    tabbed = clean_sequence("A\tC\nD E F G H I K L")
    assert plain.cleaned == trailing.cleaned == spaced.cleaned == tabbed.cleaned
    assert sequence_hash(plain.cleaned) == sequence_hash(trailing.cleaned) == sequence_hash(spaced.cleaned)
    assert sequence_hash(plain.cleaned) == sequence_hash(tabbed.cleaned)


def test_clean_rejects_garbage_without_replacement():
    with pytest.raises(InvalidSequenceError):
        clean_sequence("ACD*EFG")
    with pytest.raises(InvalidSequenceError):
        clean_sequence("")
    with pytest.raises(InvalidSequenceError):
        clean_sequence(None)
    with pytest.raises(InvalidSequenceError):
        clean_sequence(float("nan"))


def test_selenocysteine_u_is_never_silently_removed():
    sequence = "ACD" + "U" + "EFGHIKL" * 3
    result = clean_sequence(sequence)
    assert result.cleaned == sequence
    assert len(result.cleaned) == len(sequence)
    assert len(result.events) == 1
    assert result.events[0].symbol == "U"
    assert result.events[0].tokenizer_encoded_natively is True
    assert special_residue_occurrences(result.cleaned) == [(3, "U")]


def test_special_residue_kept_only_when_tokenizer_supports_it():
    support = {"U": True, "O": True}
    kept = clean_sequence("ACDU", "keep_if_supported", support)
    assert kept.cleaned == "ACDU"
    assert len(kept.events) == 1
    assert kept.events[0].tokenizer_encoded_natively is True
    with pytest.raises(UnsupportedResidueError):
        clean_sequence("ACDU", "keep_if_supported", {"U": False})
    forced = clean_sequence("ACDO", "replace_with_X", {"O": True})
    assert forced.cleaned == "ACDO"
    assert len(forced.events) == 1
    assert forced.events[0].tokenizer_encoded_natively is True
    forced_all = clean_sequence("ACDO", "replace_with_X", {"O": False})
    assert forced_all.cleaned == "ACDX"


def test_special_residue_policy_error_raises_row_level_error():
    with pytest.raises(UnsupportedResidueError) as excinfo:
        clean_sequence("ACDU", "error", {"U": False})
    assert "U" in str(excinfo.value)


def test_unknown_policy_is_rejected():
    with pytest.raises(UnsupportedResidueError):
        clean_sequence("ACDU", "remove", {"U": True})


def test_ambiguity_symbols_are_not_equivalent_and_not_dropped():
    result = clean_sequence("BZJUO")
    assert result.cleaned == "BZJUO"
    assert len(result.cleaned) == 5


# ---------------------------------------------------------------------------
# declared-length policy
# ---------------------------------------------------------------------------

def test_length_mismatch_is_a_warning_not_a_failure():
    declared, mismatch, warning = validate_declared_length(650, 649)
    assert (declared, mismatch) == (650, True)
    assert warning and "mismatch" in warning


def test_matching_and_missing_declared_length():
    assert validate_declared_length("649", 649) == (649, False, None)
    declared, mismatch, warning = validate_declared_length(None, 100)
    assert declared is None and mismatch is False and warning
    declared, mismatch, warning = validate_declared_length("abc", 100)
    assert declared is None and mismatch is False and warning


# ---------------------------------------------------------------------------
# capacity
# ---------------------------------------------------------------------------

def test_dynamic_capacity_uses_measured_special_tokens():
    assert derive_residue_capacity(2048, 2) == 2046
    assert derive_residue_capacity(1024, 1) == 1023
    with pytest.raises(CapacityError):
        derive_residue_capacity(2, 4)


def test_no_default_of_2048_when_capacity_is_absent():
    class Empty:
        pass

    with pytest.raises(CapacityError) as excinfo:
        resolve_model_max_positions(Empty())
    assert "2048" in str(excinfo.value) and "refuses" in str(excinfo.value).lower()


def test_unlimited_sentinel_model_max_length_is_rejected():
    """ESMC-600M-hf ships model_max_length ~= 1e30, which is NOT a capacity."""

    class Config:
        max_position_embeddings = 2048

    class Tokenizer:
        model_max_length = 1000000000000000019884624838656
        init_kwargs = {"model_max_length": 1000000000000000019884624838656}

    assert resolve_model_max_positions(Config(), Tokenizer()) == 2048

    class ConfigNoLimit:
        pass

    with pytest.raises(CapacityError) as excinfo:
        resolve_model_max_positions(ConfigNoLimit(), Tokenizer())
    assert "sentinel" in str(excinfo.value)


def test_chunk_size_auto_and_upper_bound():
    assert resolve_chunk_size("auto", 2046) == 2046
    assert resolve_chunk_size(512, 2046) == 512
    with pytest.raises(CapacityError) as excinfo:
        resolve_chunk_size(2048, 2046)
    assert "residue capacity" in str(excinfo.value)


# ---------------------------------------------------------------------------
# chunk planner invariants
# ---------------------------------------------------------------------------

def test_chunk_planner_invariants_for_required_lengths():
    """Planner invariants at L == capacity, capacity+1, 2439, 2347, 2286 and more."""
    for length in (CAPACITY, CAPACITY + 1, 2439, 2347, 2286, 1, 2, 257):
        for chunk_size, overlap in ((CAPACITY, 256), (512, 128), (512, 0), (100, 0), (2046, 2045)):
            if overlap >= chunk_size:
                continue
            intervals = plan_chunks(length, chunk_size, overlap)
            validate_chunk_plan(intervals, length, chunk_size)
            assert intervals[0][0] == 0
            assert intervals[-1][1] == length
            assert len({tuple(i) for i in intervals}) == len(intervals)
            assert all(0 <= s < e <= length for s, e in intervals)
            assert all(e - s <= chunk_size for s, e in intervals)
            covered = {p for s, e in intervals for p in range(s, e)}
            assert covered == set(range(length))
            assert intervals == sorted(intervals)


def test_chunk_planner_anchors_final_chunk_and_never_degenerates():
    intervals = plan_chunks(2439, 2046, 256)
    assert intervals[0] == (0, 2046)
    assert intervals[-1] == (1790, 2439)
    assert len(intervals) == 2
    assert intervals[-1][1] - intervals[-1][0] == 649
    assert intervals[-1] != intervals[0]


def test_chunk_planner_exact_fit_is_single_chunk():
    assert plan_chunks(2046, 2046, 256) == [(0, 2046)]
    assert plan_chunks(2047, 2046, 256) == [(0, 2046), (1790, 2047)]


def test_overlap_zero_uses_residue_count_weighted_chunks():
    intervals = plan_chunks(2439, 2046, 0)
    assert intervals == [(0, 2046), (2046, 2439)]
    assert sum(e - s for s, e in intervals) == 2439


def test_chunk_planner_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        plan_chunks(100, 50, 50)
    with pytest.raises(ValueError):
        plan_chunks(100, 50, 60)


def test_chunk_plan_validator_catches_gaps():
    with pytest.raises(ValueError):
        validate_chunk_plan([(0, 100), (200, 300)], 300, 100)


def test_chunk_planner_prompt12_edge_cases():
    chunk_size = 2046
    overlap = 256
    stride = chunk_size - overlap # 1790

    test_lengths = [
        chunk_size,            # 2046
        chunk_size + 1,        # 2047
        3836,                  # stride boundary 1790 + 2046
        3837,                  # 3836 + 1
        2286,
        2347,
        2439,
    ]

    for L in test_lengths:
        intervals = plan_chunks(L, chunk_size, overlap)
        validate_chunk_plan(intervals, L, chunk_size)
        assert intervals == sorted(intervals)
        assert len(intervals) == len(set(intervals))
        assert intervals[0][0] == 0
        assert intervals[-1][1] == L
        
        # Check no chunk is fully contained in another
        for i in range(len(intervals)):
            s1, e1 = intervals[i]
            assert s1 < e1
            for j in range(len(intervals)):
                if i != j:
                    s2, e2 = intervals[j]
                    assert not (s2 <= s1 and e1 <= e2), f"chunk [{s1}, {e1}) is contained in [{s2}, {e2})"

        # Check coverage
        covered = set()
        for s, e in intervals:
            covered.update(range(s, e))
        assert covered == set(range(L))

