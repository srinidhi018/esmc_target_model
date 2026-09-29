"""Sequence cleaning, hashing, special-residue policy and chunk planning.

Design rules enforced here (Section 6/8 of the specification):

* The **cleaned sequence is authoritative** for all tokenization, chunking and
  pooling. The input ``sequence_length`` column is metadata only.
* No bad sequence is ever replaced by random/zero/placeholder values, and no
  residue is ever silently dropped.
* Non-canonical symbols (``X B Z J U O``) are not garbage; the policy is
  configurable and every event is recorded.
* Chunk size is *derived* from the model's real position capacity minus the
  measured number of special tokens the tokenizer adds. ``2048`` is never
  hardcoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import CapacityError, InvalidSequenceError, UnsupportedResidueError
from .utils import sha256_text

STANDARD_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_RESIDUE_SET = frozenset(STANDARD_RESIDUES)
#: Non-canonical but chemically meaningful / conventional ambiguity symbols.
SPECIAL_RESIDUES = "XBZJUO"
SPECIAL_RESIDUE_SET = frozenset(SPECIAL_RESIDUES)
ALLOWED_RESIDUE_SET = STANDARD_RESIDUE_SET | SPECIAL_RESIDUE_SET

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class SpecialResidueEvent:
    """One transformation applied to one position of one sequence."""

    symbol: str
    position: int  # 0-based index in the CLEANED sequence
    original_symbol: str
    policy: str
    replaced_with: Optional[str] = None
    tokenizer_encoded_natively: bool = False

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "symbol": self.symbol,
            "position": self.position,
            "original_symbol": self.original_symbol,
            "policy": self.policy,
            "replaced_with": self.replaced_with,
            "tokenizer_encoded_natively": self.tokenizer_encoded_natively,
        }
        return payload


@dataclass
class CleanResult:
    cleaned: str
    original: str
    events: List[SpecialResidueEvent] = field(default_factory=list)
    warning: Optional[str] = None

    @property
    def length(self) -> int:
        return len(self.cleaned)

    @property
    def has_events(self) -> bool:
        return bool(self.events)

    def events_payload(self) -> List[Dict[str, object]]:
        return [event.to_dict() for event in self.events]


def clean_sequence(raw: object, policy: str = "keep_if_supported",
                   tokenizer_support: Optional[Dict[str, bool]] = None) -> CleanResult:
    """Clean one sequence.

    Steps: stringify, remove **all** whitespace, uppercase, then validate each
    character. Under ``keep_if_supported`` a special symbol passes through
    unchanged *provided the loaded tokenizer/model natively encodes it*; if not,
    it is replaced by ``X`` and the substitution is recorded.
    """
    if raw is None:
        raise InvalidSequenceError("sequence is missing (null/NaN)")
    if policy not in ("keep_if_supported", "error", "replace_with_X"):
        raise UnsupportedResidueError(
            f"unsupported sequence.special_residue_policy {policy!r}; the policies are exactly "
            f"'keep_if_supported', 'error' and 'replace_with_X'. A 'remove' policy is deliberately "
            f"not implemented: deleting a residue shifts every downstream position and changes the "
            f"sequence in a biologically meaningful way."
        )
    if isinstance(raw, float) and raw != raw:  # NaN
        raise InvalidSequenceError("sequence is NaN")
    text = _WHITESPACE_RE.sub("", str(raw))
    if not text:
        raise InvalidSequenceError("sequence is empty after whitespace removal")
    upper = text.upper()

    events: List[SpecialResidueEvent] = []
    cleaned_chars: List[str] = []
    for position, char in enumerate(upper):
        if char in STANDARD_RESIDUE_SET:
            cleaned_chars.append(char)
            continue
        if char in SPECIAL_RESIDUE_SET:
            supported = True if tokenizer_support is None else bool(tokenizer_support.get(char, False))
            if supported:
                cleaned_chars.append(char)
                events.append(SpecialResidueEvent(symbol=char, position=position,
                                                  original_symbol=char, policy=policy,
                                                  replaced_with=None,
                                                  tokenizer_encoded_natively=True))
                continue
            # Not natively supported by the tokenizer/model.
            if policy == "error":
                raise UnsupportedResidueError(
                    f"special residue '{char}' at position {position} is present and sequence.special_residue_policy == 'error'"
                )
            if policy == "keep_if_supported":
                raise UnsupportedResidueError(
                    f"special residue '{char}' at position {position} is not supported by the loaded "
                    f"ESMC tokenizer and sequence.special_residue_policy == 'keep_if_supported' (never replaces with X)."
                )
            if char == "X":
                raise UnsupportedResidueError(
                    f"special residue 'X' at position {position} is not supported by the loaded "
                    f"ESMC tokenizer and cannot be replaced with itself."
                )
            if policy == "replace_with_X":
                cleaned_chars.append("X")
                events.append(SpecialResidueEvent(symbol=char, position=position,
                                                  original_symbol=char, policy=policy,
                                                  replaced_with="X",
                                                  tokenizer_encoded_natively=False))
                continue
            raise UnsupportedResidueError(
                f"unknown sequence.special_residue_policy {policy!r} for residue '{char}'"
            )
        raise InvalidSequenceError(
            f"invalid character {char!r} at position {position}: not a standard amino acid "
            f"(ACDEFGHIKLMNPQRSTVWY) nor a recognised non-canonical symbol (XBZJUO)"
        )

    cleaned = "".join(cleaned_chars)
    if not cleaned:
        raise InvalidSequenceError("sequence is empty after validation")
    return CleanResult(cleaned=cleaned, original=str(raw), events=events)


def sequence_hash(cleaned_sequence: str) -> str:
    """SHA256 of the cleaned sequence: the only protein identity key used here."""
    return sha256_text(cleaned_sequence)


def validate_declared_length(declared: object, actual: int) -> Tuple[Optional[int], bool, Optional[str]]:
    """Compare the metadata ``sequence_length`` against the cleaned length.

    Returns ``(declared_or_None, mismatch_flag, warning_or_None)``. A mismatch is
    a **metadata discrepancy**, not a failure: it never sends a row to
    ``failed_rows.csv``.
    """
    if declared is None or (isinstance(declared, float) and declared != declared):
        return None, False, "input sequence_length is missing/invalid (warning only; cleaned sequence is authoritative)"
    try:
        declared_int = int(declared)
    except (TypeError, ValueError):
        return None, False, f"input sequence_length {declared!r} is not an integer (warning only)"
    if declared_int != actual:
        return declared_int, True, (
            f"sequence_length metadata mismatch: declared={declared_int}, actual={actual} "
            f"(warning only; the cleaned sequence is authoritative)"
        )
    return declared_int, False, None


def special_residue_occurrences(sequence: str) -> List[Tuple[int, str]]:
    """(position, symbol) for every non-standard character, in order."""
    return [(i, ch) for i, ch in enumerate(sequence) if ch in SPECIAL_RESIDUE_SET]


# ---------------------------------------------------------------------------
# Capacity + chunk planning
# ---------------------------------------------------------------------------

def derive_special_token_count(tokenizer: object) -> int:
    """Measure how many tokens the tokenizer *adds* around a sequence.

    Authoritative: requires return_special_tokens_mask=True from the tokenizer.
    Does NOT infer or guess special-token count from CLS/BOS/EOS/SEP token IDs.
    """
    encode = getattr(tokenizer, "__call__", None)
    if encode is not None:
        try:
            out = tokenizer("M", return_special_tokens_mask=True, return_attention_mask=True, truncation=False)
        except TypeError:
            try:
                out = tokenizer("M", return_tensors=None)
            except Exception:
                out = tokenizer("M")
        ids = _first_field(out, "input_ids")
        mask = _first_field(out, "special_tokens_mask")
        if ids is not None and mask is not None and len(mask) == len(ids):
            return int(sum(1 for m in mask if m == 1))
    raise CapacityError(
        "Could not measure how many special tokens the tokenizer adds to a sequence: "
        "the tokenizer did not return a valid special_tokens_mask. Refusing to guess a capacity."
    )


def _first_field(obj: object, key: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _plausible_max_length(value: Any) -> Optional[int]:
    """Accept only a plausible position limit.

    Some checkpoints (including ESMC-600M-hf) ship a ``model_max_length`` of
    ~1e30, which is a "no limit" sentinel rather than a real capacity. Such a
    value is rejected instead of being used to derive a chunk size.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value <= 0 or value > 1_000_000:
        return None
    return int(value)


def resolve_model_max_positions(model_config: object, tokenizer: object = None) -> int:
    """Authoritative maximum number of token positions the model accepts.

    Order: explicit override (handled by the caller) -> ``max_position_embeddings``
    and friends on the model config -> tokenizer/model init kwargs. Values that
    are obvious "unlimited" sentinels are rejected. There is deliberately **no**
    ``getattr(..., 2048)`` fallback.
    """
    candidates = ("max_position_embeddings", "n_positions", "max_seq_len", "max_sequence_length")
    for name in candidates:
        value = _plausible_max_length(getattr(model_config, name, None))
        if value is not None:
            return value
    for holder in (tokenizer, model_config):
        init_kwargs = getattr(holder, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            for name in candidates:
                value = _plausible_max_length(init_kwargs.get(name))
                if value is not None:
                    return value
    raise CapacityError(
        "Could not establish an authoritative maximum sequence length: the model config exposes "
        "none of max_position_embeddings / n_positions / max_seq_len / max_sequence_length, and "
        "the tokenizer's model_max_length is either absent or an 'unlimited' sentinel. This "
        "pipeline refuses to default to 2048. Inspect the checkpoint config and set "
        "sequence.max_model_positions explicitly."
    )


def derive_residue_capacity(model_max_positions: int, special_tokens: int) -> int:
    """residues per chunk = model positions - tokens the tokenizer adds.

    Guarantees, for every chunk actually sent to the model:
    ``residues_in_chunk + special_tokens <= model_max_positions``.
    """
    if special_tokens < 0:
        raise CapacityError(f"special token count must be >= 0, got {special_tokens}")
    capacity = int(model_max_positions) - int(special_tokens)
    if capacity < 1:
        raise CapacityError(
            f"Derived residue capacity is {capacity}: model_max_positions={model_max_positions} "
            f"minus {special_tokens} special token(s). The model cannot accept a single residue; "
            f"the checkpoint is not usable for protein encoding."
        )
    return capacity


def resolve_chunk_size(configured: object, residue_capacity: int) -> int:
    if configured in (None, "auto"):
        return int(residue_capacity)
    if not isinstance(configured, int):
        raise CapacityError(f"sequence.chunk_size must be 'auto' or an int, got {configured!r}")
    if configured > residue_capacity:
        raise CapacityError(
            f"sequence.chunk_size={configured} exceeds the derived residue capacity "
            f"{residue_capacity} (= model_max_positions - measured special tokens). "
            f"Lower chunk_size or raise the model's position limit; never send more positions "
            f"than the model accepts."
        )
    return int(configured)


def plan_chunks(length: int, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    """Deterministic, ordered, non-duplicated ``[start, end)`` residue intervals.

    Invariants (Section 3):
    * ``0 <= start < end <= L``; ``end - start <= chunk_size``; no empty chunks.
    * The first interval starts at 0; the final interval ends exactly at ``L``.
    * ``stride = chunk_size - overlap`` with ``overlap < chunk_size``.
    * Adjacent chunks step by stride = chunk_size - overlap.
    * The final chunk may be shorter than chunk_size (end = min(start + chunk_size, L)).
    * No artificial backward shifting of the final chunk to create enormous overlap.
    * The union of intervals covers every residue position without gaps.
    """
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly smaller than chunk_size ({chunk_size})"
        )
    if length == 0:
        return []
    if length <= chunk_size:
        return [(0, length)]

    stride = chunk_size - overlap
    intervals: List[Tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        intervals.append((start, end))
        if end == length:
            break
        start += stride

    validate_chunk_plan(intervals, length, chunk_size)
    return intervals


def validate_chunk_plan(intervals: Sequence[Tuple[int, int]], length: int, chunk_size: int) -> None:
    if not intervals and length > 0:
        raise ValueError("chunk plan is empty for a non-empty sequence")
    if intervals[0][0] != 0:
        raise ValueError(f"first chunk must start at residue 0, got {intervals[0]}")
    if intervals[-1][1] != length:
        raise ValueError(f"final chunk must end at residue {length}, got {intervals[-1]}")
    seen = set()
    covered = set()
    for start, end in intervals:
        if start < 0 or end > length:
            raise ValueError(f"chunk [{start}, {end}) is outside [0, {length})")
        if end <= start:
            raise ValueError(f"empty chunk [{start}, {end})")
        if end - start > chunk_size:
            raise ValueError(f"chunk [{start}, {end}) exceeds chunk_size {chunk_size}")
        if (start, end) in seen:
            raise ValueError(f"duplicate chunk interval {start, end}")
        seen.add((start, end))
        covered.update(range(start, end))
    missing = sorted(set(range(length)) - covered)
    if missing:
        raise ValueError(
            f"chunk plan leaves {len(missing)} residue position(s) uncovered "
            f"(first missing: {missing[0]})"
        )
