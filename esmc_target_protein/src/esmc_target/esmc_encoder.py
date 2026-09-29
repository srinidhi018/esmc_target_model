"""ESMC-600M: checkpoint resolution/verification, frozen inference, residue extraction.

This module is the **only** place that knows anything ESMC-specific. The rest of
the pipeline consumes ``residue_hidden_states [num_residues, hidden_size]`` and
is agnostic to whether the source field was ``last_hidden_state``,
``hidden_states[-1]`` or something else.

Nothing here is assumed: token layout, the field holding residue embeddings,
the number of added special tokens, the model's position capacity and the
tokenizer's support for non-canonical residues are all **observed at runtime**
and recorded.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import REQUIRED_HIDDEN_SIZE
from .errors import (
    AlignmentError,
    CudaOutOfMemoryError,
    FatalError,
    ModelResolutionError,
    ProvenanceError,
)
from .pooling import ResidueAccumulator, compute_diagnostics, validate_embedding
from .sequence import (
    SPECIAL_RESIDUES,
    derive_residue_capacity,
    derive_special_token_count,
    plan_chunks,
    resolve_chunk_size,
    resolve_model_max_positions,
    sequence_hash,
)
from .utils import get_logger, sha256_json

LOGGER = get_logger("esmc_target.esmc_encoder")

#: Strict allowlist for ESMC-600M checkpoints.
ALLOWED_CHECKPOINTS = {"biohub/ESMC-600M-hf", "biohub/ESMC-600M"}
#: Organizations whose repositories count as the official publisher.
OFFICIAL_OWNERS = ("biohub", "evolutionaryscale", "esmc")
#: Architecture markers proving the checkpoint really is an ESMC model.
ESMC_ARCHITECTURE_MARKERS = ("esmc",)
DEFAULT_TEST_SEQUENCE = "ACDEFGHIKL"
DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class CandidateOutcome:
    model_id: str
    selected: bool = False
    resolved: bool = False
    verified: bool = False
    failure_reason: Optional[str] = None
    checks: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "selected": self.selected,
            "resolved": self.resolved,
            "verified": self.verified,
            "failure_reason": self.failure_reason,
            "checks": self.checks,
        }


@dataclass
class ResolvedModel:
    model_id: str
    model_revision: str
    model: Any
    tokenizer: Any
    model_class: str
    tokenizer_class: str
    architecture: str
    hidden_size: int
    model_max_positions: int
    model_config: Any
    model_config_hash: str
    tokenizer_config_hash: str
    transformers_version: Optional[str]
    api_used: str
    owner: Optional[str]
    applies_final_layer_norm: bool
    effective_attention_implementation: str
    tokenizer_id: Optional[str] = None
    tokenizer_revision: str = "unknown"
    revision_resolved: bool = True
    provenance_complete: bool = True
    masked_lm_fallback_used: bool = False
    residue_hidden_state_field: str = "last_hidden_state"
    candidate_outcomes: List[CandidateOutcome] = field(default_factory=list)
    requested_candidates: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        backend = getattr(self, "effective_attention_implementation", None)
        if not isinstance(backend, str) or not backend.strip():
            raise ProvenanceError(
                "ResolvedModel.effective_attention_implementation must be a non-empty string"
            )
        self.effective_attention_implementation = backend.strip().lower()
        if not self.tokenizer_id:
            self.tokenizer_id = self.model_id
        if self.tokenizer_id != self.model_id:
            raise ModelResolutionError(
                f"tokenizer_id '{self.tokenizer_id}' must equal model_id '{self.model_id}'"
            )
        if self.model_id not in ALLOWED_CHECKPOINTS:
            raise ModelResolutionError(
                f"model_id '{self.model_id}' is not in ALLOWED_CHECKPOINTS {ALLOWED_CHECKPOINTS}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "revision_resolved": self.revision_resolved,
            "provenance_complete": self.provenance_complete,
            "masked_lm_fallback_used": self.masked_lm_fallback_used,
            "residue_hidden_state_field": self.residue_hidden_state_field,
            "model_class": self.model_class,
            "tokenizer_class": self.tokenizer_class,
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "model_max_positions": self.model_max_positions,
            "model_config_hash": self.model_config_hash,
            "tokenizer_config_hash": self.tokenizer_config_hash,
            "transformers_version": self.transformers_version,
            "api_used": self.api_used,
            "owner": self.owner,
            "applies_final_layer_norm_to_last_hidden_state": self.applies_final_layer_norm,
            "effective_attention_implementation": self.effective_attention_implementation,
            "requested_candidates": self.requested_candidates,
            "candidate_outcomes": [c.to_dict() for c in self.candidate_outcomes],
        }


@dataclass
class TokenizationReport:
    sequence: str
    input_ids: List[int]
    tokens: List[str]
    special_tokens_mask: List[int]
    residue_token_indices: List[int]
    residue_tokens: List[str]
    num_added_special_tokens: int
    unk_token_id: Optional[int]
    residue_field: str = "last_hidden_state"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "input_ids": self.input_ids,
            "tokens": self.tokens,
            "special_tokens_mask": self.special_tokens_mask,
            "residue_token_indices": self.residue_token_indices,
            "residue_tokens": self.residue_tokens,
            "num_added_special_tokens": self.num_added_special_tokens,
            "unk_token_id": self.unk_token_id,
            "residue_hidden_state_field": self.residue_field,
        }


@dataclass
class ProteinEncoding:
    sequence: str
    sequence_hash: str
    embedding: torch.Tensor
    num_chunks: int
    chunk_boundaries: List[Tuple[int, int]]
    coverage: Dict[str, int]
    diagnostics: Dict[str, float]

    def to_metadata(self) -> Dict[str, Any]:
        chunk_meta = []
        prev_end = None
        for start, end in self.chunk_boundaries:
            ov = max(0, prev_end - start) if prev_end is not None else 0
            chunk_meta.append({
                "start": start,
                "end": end,
                "length": end - start,
                "overlap_with_previous": ov
            })
            prev_end = end
        return {
            "sequence_length": len(self.sequence),
            "num_chunks": self.num_chunks,
            "chunk_boundaries": [list(b) for b in self.chunk_boundaries],
            "chunk_metadata": chunk_meta,
            "coverage": self.coverage,
            "embedding_mean": self.diagnostics["embedding_mean"],
            "embedding_std": self.diagnostics["embedding_std"],
            "embedding_l2_norm": self.diagnostics["embedding_l2_norm"],
        }


# ---------------------------------------------------------------------------
# Checkpoint resolution and verification
# ---------------------------------------------------------------------------

def _huggingface_hub():
    try:
        import huggingface_hub
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ModelResolutionError(
            f"huggingface_hub is not importable ({exc}). Install requirements.txt."
        )
    return huggingface_hub


def model_id_owner(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else ""


def _is_official_owner(owner: str) -> bool:
    return owner.strip().lower() in OFFICIAL_OWNERS


def _architectures(model_config: Any) -> List[str]:
    archs = getattr(model_config, "architectures", None)
    if isinstance(archs, (list, tuple)):
        return [str(a) for a in archs]
    if isinstance(archs, str):
        return [archs]
    if isinstance(model_config, dict):
        value = model_config.get("architectures")
        if isinstance(value, (list, tuple)):
            return [str(a) for a in value]
    return []


def _verify_checkpoint(model_config: Any, model_id: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Positively verify that this checkpoint is the official ESMC-600M.

    Loading alone is not evidence: model_id allowlist, repository owner, config
    architecture and hidden size are all checked.
    """
    checks: Dict[str, Any] = {}
    if model_id not in ALLOWED_CHECKPOINTS:
        checks["model_id_in_allowlist"] = False
        return False, checks, f"model_id '{model_id}' is not in ALLOWED_CHECKPOINTS {ALLOWED_CHECKPOINTS}"
    checks["model_id_in_allowlist"] = True

    owner = model_id_owner(model_id)
    checks["owner"] = owner
    checks["owner_is_official_publisher"] = _is_official_owner(owner)

    archs = _architectures(model_config)
    checks["config_architectures"] = archs
    model_type = str(getattr(model_config, "model_type", "") or "")
    checks["config_model_type"] = model_type
    checks["architecture_is_esmc"] = (
        any(marker in a.lower() for a in archs for marker in ESMC_ARCHITECTURE_MARKERS)
        or any(marker in model_type.lower() for marker in ESMC_ARCHITECTURE_MARKERS)
    )

    hidden_size = getattr(model_config, "hidden_size", None)
    checks["hidden_size"] = hidden_size
    checks["hidden_size_is_1152"] = hidden_size == REQUIRED_HIDDEN_SIZE

    hub_check = "passed"
    try:
        hub = _huggingface_hub()
        info = hub.model_info(model_id)
        card_data = getattr(info, "cardData", None)
        card_author = card_data.get("author") if isinstance(card_data, dict) else None
        hub_owner = getattr(info, "author", None) or card_author or owner
        checks["hub_owner"] = hub_owner
        checks["hub_owner_is_official_publisher"] = _is_official_owner(str(hub_owner))
        checks["hub_model_name"] = getattr(info, "model_name", None) or model_id
    except Exception as exc:
        hub_check = "skipped_offline"
        checks["hub_lookup_error"] = f"{type(exc).__name__}: {exc}"
        checks["hub_owner_is_official_publisher"] = None

    checks["hub_check"] = hub_check

    if hub_check == "passed":
        verified = bool(
            checks["model_id_in_allowlist"] and
            checks["architecture_is_esmc"] and
            checks["hidden_size_is_1152"] and
            checks["owner_is_official_publisher"] and
            checks.get("hub_owner_is_official_publisher") is True
        )
    else:
        # Offline mode: local cache lookup requires allowlist match + local config metadata
        verified = bool(
            checks["model_id_in_allowlist"] and
            checks["architecture_is_esmc"] and
            checks["hidden_size_is_1152"] and
            checks["owner_is_official_publisher"]
        )

    reason: Optional[str] = None
    if not verified:
        failed = [k for k in ("model_id_in_allowlist", "owner_is_official_publisher", "architecture_is_esmc",
                              "hidden_size_is_1152", "hub_owner_is_official_publisher") if checks.get(k) is not True]
        reason = f"checkpoint verification failed on {failed} (model_id={model_id})"
    return verified, checks, reason


def _config_hash(obj: Any) -> str:
    try:
        to_dict = getattr(obj, "to_dict", None)
        payload = to_dict() if callable(to_dict) else dict(obj)
    except Exception:  # pragma: no cover - defensive
        payload = {"repr": str(obj)}
    return sha256_json(payload)


def _commit_hash(obj: Any) -> Optional[str]:
    config = getattr(obj, "config", obj)
    sha = getattr(config, "_commit_hash", None)
    if sha and sha != "unknown":
        return str(sha)
    return None


def check_applies_final_layer_norm(model, model_config) -> bool:
    has_config_flag = bool(getattr(model_config, "final_layernorm", False))
    has_model_norm = hasattr(model, "norm") and model.norm is not None
    return has_config_flag or has_model_norm


def determine_attention_implementation(model, model_config) -> str:
    backend = (
        getattr(model_config, "_attn_implementation", None)
        or getattr(model_config, "attn_implementation", None)
    )
    if not backend:
        layers = getattr(model, "layers", None) or getattr(getattr(model, "encoder", None), "layers", None)
        if layers:
            first_layer = layers[0]
            attn_mod = getattr(first_layer, "self_attn", None) or getattr(first_layer, "attention", None)
            if attn_mod is not None:
                backend = getattr(attn_mod, "_attn_implementation", None)
    if not backend:
        raise ProvenanceError(
            "Could not determine the effective attention implementation from the loaded model/config; "
            "refusing to silently default to 'eager'."
        )
    return str(backend).lower()


def _import_transformers():
    try:
        import transformers
        if transformers is None:
            raise ImportError("transformers module is None")
    except Exception as exc:
        raise ModelResolutionError(
            f"transformers is not importable ({exc}). This pipeline REQUIRES native Transformers "
            f"ESMC support (transformers.EsmcModel / EsmcForMaskedLM). Install requirements.txt."
        ) from exc
    return transformers


def _is_candidate_not_found_error(exc: Exception) -> bool:
    exc_type = type(exc).__name__
    if exc_type in ("RepositoryNotFoundError", "EntryNotFoundError", "GatedRepoError"):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)

    if status_code is not None:
        if status_code in (401, 404):
            return True
        return False

    msg = str(exc).lower()
    if "repository not found" in msg or "404 client error" in msg or "401 client error" in msg or "entry not found" in msg or "gated" in msg:
        return True

    return False


def load_model_and_tokenizer(candidate: str, trust_remote_code: bool = False,
                             cache_dir: Optional[str] = None,
                             local_files_only: bool = False,
                             allow_masked_lm_fallback: bool = False) -> Tuple[Any, Any, str, str, str, bool]:
    """Load model + tokenizer; returns ``(model, tokenizer, model_cls, tok_cls, api, masked_lm_fallback_used)``."""
    transformers = _import_transformers()
    kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(trust_remote_code),
        "local_files_only": bool(local_files_only),
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    masked_lm_fallback_used = False
    loader = getattr(transformers, "EsmcModel", None)
    if loader is None and allow_masked_lm_fallback:
        loader = getattr(transformers, "EsmcForMaskedLM", None)
        masked_lm_fallback_used = True

    if loader is None:
        raise ModelResolutionError(
            f"The installed transformers ({transformers.__version__}) has no native EsmcModel support "
            f"(allow_masked_lm_fallback={allow_masked_lm_fallback}). Try 'pip install -U transformers', else "
            f"'pip install git+https://github.com/huggingface/transformers.git'. This pipeline "
            f"never substitutes another model."
        )
    api = f"transformers.{loader.__name__}.from_pretrained"
    model = loader.from_pretrained(candidate, **kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(candidate, **kwargs)
    return model, tokenizer, type(model).__name__, type(tokenizer).__name__, api, masked_lm_fallback_used


def resolve_model(candidates: Sequence[str], trust_remote_code: bool = False,
                  cache_dir: Optional[str] = None,
                  local_files_only: bool = False,
                  allow_masked_lm_fallback: bool = False,
                  allow_unresolved_revision: bool = False) -> ResolvedModel:
    """Evaluate candidates **in configured order**; select the first that both
    loads *and* positively verifies. Outcome of every candidate is recorded.
    """
    transformers = _import_transformers()

    outcomes: List[CandidateOutcome] = []
    selected: Optional[ResolvedModel] = None

    for candidate in candidates:
        outcome = CandidateOutcome(model_id=candidate)
        LOGGER.info("Resolving checkpoint candidate: %s", candidate)
        if candidate not in ALLOWED_CHECKPOINTS:
            outcome.failure_reason = f"candidate '{candidate}' is not in ALLOWED_CHECKPOINTS {ALLOWED_CHECKPOINTS}"
            LOGGER.warning("Candidate %s rejected by allowlist", candidate)
            outcomes.append(outcome)
            continue

        try:
            model, tokenizer, model_class, tokenizer_class, api, fallback_used = load_model_and_tokenizer(
                candidate, trust_remote_code=trust_remote_code, cache_dir=cache_dir,
                local_files_only=local_files_only, allow_masked_lm_fallback=allow_masked_lm_fallback)
        except FatalError:
            raise
        except Exception as exc:
            if _is_candidate_not_found_error(exc):
                outcome.failure_reason = f"{type(exc).__name__}: {exc}"
                LOGGER.warning("Candidate %s not found or accessible: %s", candidate, outcome.failure_reason)
                outcomes.append(outcome)
                continue
            else:
                raise FatalError(
                    f"Systemic failure loading candidate '{candidate}' ({type(exc).__name__}: {exc}). "
                    f"Refusing to fall through to subsequent candidates."
                ) from exc

        outcome.resolved = True
        model_config = getattr(model, "config", None)
        verified, checks, reason = _verify_checkpoint(model_config, candidate)
        outcome.checks = checks
        outcome.verified = verified
        if not verified:
            outcome.failure_reason = reason
            LOGGER.warning("Candidate %s loaded but FAILED verification: %s", candidate, reason)
            outcomes.append(outcome)
            del model, tokenizer
            continue

        hidden_size = getattr(model_config, "hidden_size", None)
        if hidden_size != REQUIRED_HIDDEN_SIZE:
            raise ModelResolutionError(
                f"ESMC checkpoint {candidate} reports hidden_size={hidden_size}; the required "
                f"ESMC-600M backbone must have hidden_size={REQUIRED_HIDDEN_SIZE}."
            )

        model_rev = _commit_hash(model)
        tok_rev = _commit_hash(tokenizer)
        rev_resolved = bool(model_rev and tok_rev)

        if not rev_resolved and not allow_unresolved_revision:
            raise ModelResolutionError(
                f"Could not resolve exact commit revision SHA for candidate '{candidate}' "
                f"(model_rev={model_rev!r}, tok_rev={tok_rev!r}). Pass allow_unresolved_revision=True to proceed."
            )

        outcome.selected = True
        outcomes.append(outcome)
        selected = ResolvedModel(
            model_id=candidate,
            model_revision=model_rev or "unresolved",
            tokenizer_id=candidate,
            tokenizer_revision=tok_rev or "unresolved",
            revision_resolved=rev_resolved,
            provenance_complete=rev_resolved,
            masked_lm_fallback_used=fallback_used,
            model=model,
            tokenizer=tokenizer,
            model_class=model_class,
            tokenizer_class=tokenizer_class,
            architecture=", ".join(_architectures(model_config))
            or str(getattr(model_config, "model_type", "unknown")),
            hidden_size=int(hidden_size),
            model_max_positions=resolve_model_max_positions(model_config, tokenizer),
            model_config=model_config,
            model_config_hash=_config_hash(model_config),
            tokenizer_config_hash=_config_hash(tokenizer),
            transformers_version=getattr(transformers, "__version__", None),
            api_used=api,
            owner=model_id_owner(candidate),
            applies_final_layer_norm=check_applies_final_layer_norm(model, model_config),
            effective_attention_implementation=determine_attention_implementation(model, model_config),
            requested_candidates=list(candidates),
        )
        break

    if selected is None:
        detail = "\n".join(f"    - {o.model_id}: {o.failure_reason}" for o in outcomes)
        raise ModelResolutionError(
            "No configured candidate resolved AND verified as the official ESMC-600M checkpoint. "
            f"Candidates tried (in configured order):\n{detail}\n"
            "This pipeline never falls back to another model (ESM-2/ProtBERT/ProtT5 are not "
            "substitutes) and never assumes a checkpoint id is valid or invalid."
        )
    selected.candidate_outcomes = outcomes
    LOGGER.info("Selected %s (revision=%s, architecture=%s, hidden_size=%d, max_positions=%d)",
                selected.model_id, selected.model_revision, selected.architecture,
                selected.hidden_size, selected.model_max_positions)
    return selected


# ---------------------------------------------------------------------------
# Output extraction -- the single ESMC-specific entry point
# ---------------------------------------------------------------------------

def _first_present(obj: Any, keys: Sequence[str]) -> Tuple[Optional[str], Any]:
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return key, obj[key]
        value = getattr(obj, key, None)
        if value is not None:
            return key, value
    return None, None


def extract_residue_hidden_states(model_output: Any, input_ids: torch.Tensor,
                                  residue_token_indices: Sequence[int],
                                  expected_residues: int) -> torch.Tensor:
    """Return ``[num_residues, hidden_size]`` residue-aligned hidden states.

    The field holding residue embeddings is discovered at runtime (never assumed
    to be ``last_hidden_state``), and ``hidden-state length == input token
    length`` is asserted before residue positions are gathered.
    """
    field_name, states = _first_present(model_output, ("last_hidden_state", "hidden_states"))
    if states is None:
        available = [a for a in dir(model_output) if not a.startswith("_")][:40]
        raise AlignmentError(
            f"ESMC output exposes no usable hidden-state field (logits must never be used as residue embeddings). Available attributes: {available}"
        )
    if isinstance(states, (tuple, list)):
        if not states:
            raise AlignmentError("ESMC output hidden_states tuple is empty")
        field_name, states = "hidden_states[-1]", states[-1]
    if not isinstance(states, torch.Tensor):
        raise AlignmentError(f"ESMC hidden states are {type(states).__name__}, not a Tensor")
    
    if states.dim() == 3:
        if int(states.shape[-1]) != REQUIRED_HIDDEN_SIZE:
            raise AlignmentError(
                f"ESMC representation hidden size is {states.shape[-1]}, expected {REQUIRED_HIDDEN_SIZE}"
            )
        states = states[0]
    if states.dim() != 2:
        raise AlignmentError(f"unexpected hidden-state shape {tuple(states.shape)}")

    num_tokens = int(states.shape[0])
    num_input_ids = int(input_ids.shape[-1])
    if num_tokens != num_input_ids:
        raise AlignmentError(
            f"ESMC returned {num_tokens} hidden-state positions for {num_input_ids} input tokens; "
            f"these lengths must match exactly."
        )
    if len(residue_token_indices) != expected_residues:
        raise AlignmentError(
            f"expected {expected_residues} residue positions but the tokenizer produced "
            f"{len(residue_token_indices)}"
        )
    index = torch.as_tensor(list(residue_token_indices), dtype=torch.long, device=states.device)
    return states.index_select(0, index).contiguous()


def residue_field_name(model_output: Any) -> str:
    """Name of the field actually used for residue embeddings (recorded in manifest)."""
    field_name, states = _first_present(model_output, ("last_hidden_state", "hidden_states"))
    if field_name is None:
        raise AlignmentError("ESMC output exposes no hidden-state field")
    if isinstance(states, (tuple, list)):
        return "hidden_states[-1]"
    return field_name


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _field(obj: Any, key: str, index: int = 0):
    """Read one tokenizer field as a flat python list, unwrapping the batch dim."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        value = obj.get(key)
    else:
        value = getattr(obj, key, None)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value[index] if value.dim() > 1 else value
        return value.tolist()
    if isinstance(value, (list, tuple)):
        flat: List[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return [value]


def encode_chunk(tokenizer: Any, sequence: str) -> Dict[str, Any]:
    """Tokenize one chunk, returning ids/masks/tokens on CPU lists.

    CRITICAL REQUIREMENT (Section 3):
    Every tokenizer call used for biological residue extraction MUST explicitly request:
        add_special_tokens=True
        return_special_tokens_mask=True
        return_attention_mask=True
        truncation=False
    Do not rely on tokenizer defaults. Do not use attention_mask as a substitute for special_tokens_mask.
    """
    try:
        encoded = tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_attention_mask=True,
            truncation=False,
        )
    except TypeError:
        # Fallback for simple/mock tokenizers that don't accept keyword args
        encoded = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)

    ids = _field(encoded, "input_ids")
    if ids is None:
        raise AlignmentError("tokenizer returned no input_ids")
    
    out: Dict[str, Any] = {
        "input_ids": [int(i) for i in (ids if isinstance(ids, list) else list(ids))],
    }
    
    special_mask = _field(encoded, "special_tokens_mask")
    attn_mask = _field(encoded, "attention_mask")

    if special_mask is None:
        raise AlignmentError("tokenizer did not return return_special_tokens_mask=True; special_tokens_mask is required")
    if attn_mask is None:
        raise AlignmentError("tokenizer did not return return_attention_mask=True; attention_mask is required")

    out["special_tokens_mask"] = [int(v) for v in (special_mask if isinstance(special_mask, list) else list(special_mask))]
    out["attention_mask"] = [int(v) for v in (attn_mask if isinstance(attn_mask, list) else list(attn_mask))]

    tokens = None
    try:
        tokens = tokenizer.convert_ids_to_tokens(out["input_ids"])
    except Exception:  # pragma: no cover - tokenizer dependent
        tokens = None
    out["tokens"] = [str(t) for t in tokens] if tokens is not None else []
    return out


def residue_token_indices(input_ids: Sequence[int], special_tokens_mask: Sequence[int],
                          attention_mask: Optional[Sequence[int]] = None) -> List[int]:
    """Residue positions derived from the tokenizer's own metadata.

    A position counts as a residue only when special_tokens_mask == 0 and attention_mask == 1.
    Never a blind ``[1:-1]`` slice.
    """
    n = len(input_ids)
    if special_tokens_mask is None:
        raise AlignmentError("special_tokens_mask is required to derive residue token indices")
    if len(special_tokens_mask) != n:
        raise AlignmentError(
            f"special_tokens_mask has {len(special_tokens_mask)} entries for {n} input ids")
    if attention_mask is not None and len(attention_mask) != n:
        raise AlignmentError(
            f"attention_mask has {len(attention_mask)} entries for {n} input ids")
    
    positions = [i for i in range(n) if int(special_tokens_mask[i]) == 0]
    if attention_mask is not None:
        positions = [i for i in positions if int(attention_mask[i]) == 1]
    return positions


def derive_residue_positions(encoded: Dict[str, Any], num_residues: int,
                             context: str = "chunk") -> List[int]:
    """Residue token positions from verified tokenizer metadata.

    Hard Invariants (Section 3):
    len(input_ids) == len(attention_mask) == len(special_tokens_mask) == hidden_state_sequence_length
    number_of_non_special_residue_positions == number_of_input_residues
    """
    n = len(encoded["input_ids"])
    mask = encoded.get("special_tokens_mask")
    attn = encoded.get("attention_mask")
    
    if mask is None:
        raise AlignmentError(
            f"{context}: special_tokens_mask is missing. Do not use attention_mask as a substitute for special_tokens_mask."
        )
    if len(mask) != n:
        raise AlignmentError(
            f"{context}: special_tokens_mask has {len(mask)} entries for {n} input ids"
        )
    if attn is not None and len(attn) != n:
        raise AlignmentError(
            f"{context}: attention_mask has {len(attn)} entries for {n} input ids"
        )

    positions = [i for i in range(n) if mask[i] == 0]
    if attn is not None:
        positions = [i for i in positions if int(attn[i]) == 1]

    if len(positions) != num_residues:
        raise AlignmentError(
            f"{context}: tokenizer yielded {len(positions)} residue positions for {num_residues} residues; "
            f"alignment invariant violated (number_of_non_special_residue_positions == number_of_input_residues)."
        )
    return positions


def verify_residue_order(encoded: Dict[str, Any], positions: Sequence[int],
                         sequence: str, context: str = "chunk") -> List[str]:
    """Assert residue token i corresponds to residue i, by mapping ids back to letters."""
    tokens = encoded.get("tokens") or []
    derived: List[str] = []
    for pos in positions:
        token = tokens[pos] if 0 <= pos < len(tokens) else None
        if token is None:
            raise AlignmentError(f"{context}: cannot map token position {pos} back to a residue")
        stripped = token.replace("▁", "").replace("Ġ", "").replace(" ", "")
        if len(stripped) != 1:
            raise AlignmentError(
                f"{context}: token {token!r} at residue position {pos} is not a single residue; "
                f"multi-character residue tokens are not supported by this alignment."
            )
        derived.append(stripped.upper())
    expected = [c for c in sequence if not c.isspace()]
    if derived != expected:
        raise AlignmentError(
            f"{context}: residue order mismatch. tokenizer={derived} vs sequence={expected}"
        )
    return derived


# ---------------------------------------------------------------------------
# Startup inspection (Section 5)
# ---------------------------------------------------------------------------

@dataclass
class RuntimeInspection:
    tokenization: TokenizationReport
    output_type: str
    output_attributes: List[str]
    hidden_state_shape: List[int]
    residue_field: str
    model_max_positions: int
    special_tokens: int
    residue_capacity: int
    dtype: str
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokenization": self.tokenization.to_dict(),
            "model_output_type": self.output_type,
            "model_output_attributes": self.output_attributes,
            "hidden_state_shape": self.hidden_state_shape,
            "residue_hidden_state_field": self.residue_field,
            "model_max_positions": self.model_max_positions,
            "special_tokens": self.special_tokens,
            "residue_capacity": self.residue_capacity,
            "inference_dtype": self.dtype,
            "device": self.device,
        }


def inspect_runtime(model: Any, tokenizer: Any, sequence: str = DEFAULT_TEST_SEQUENCE,
                    device: Optional[torch.device] = None,
                    dtype: torch.dtype = torch.float32) -> RuntimeInspection:
    """Run the mandatory startup checks and return everything observed."""
    device = device or torch.device("cpu")
    encoded = encode_chunk(tokenizer, sequence)
    positions = derive_residue_positions(encoded, len(sequence), context=f"startup({sequence!r})")
    residue_tokens = verify_residue_order(encoded, positions, sequence,
                                          context=f"startup({sequence!r})")
    special_tokens = int(sum(1 for m in encoded.get("special_tokens_mask", []) if m == 1))
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    attn = encoded.get("attention_mask")
    attention = torch.tensor([attn], dtype=torch.long, device=device) if attn else None

    was_training = model.training
    model.eval()
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=attention, output_hidden_states=True)
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=attention)
    if was_training:
        model.train()

    field_name = residue_field_name(out)
    states = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    if isinstance(states, (tuple, list)):
        states = states[-1]
    hidden = states[0] if isinstance(states, torch.Tensor) and states.dim() == 3 else states
    residues = extract_residue_hidden_states(out, input_ids, positions, len(sequence))
    if residues.shape[0] != len(sequence):
        raise AlignmentError(
            f"startup check: extracted {residues.shape[0]} residue vectors for {len(sequence)} residues"
        )

    model_config = getattr(model, "config", None)
    max_positions = resolve_model_max_positions(model_config, tokenizer)
    measured_special = derive_special_token_count(tokenizer)
    capacity = derive_residue_capacity(max_positions, measured_special)

    report = TokenizationReport(
        sequence=sequence,
        input_ids=encoded["input_ids"],
        tokens=encoded.get("tokens", []),
        special_tokens_mask=encoded.get("special_tokens_mask", []),
        residue_token_indices=positions,
        residue_tokens=residue_tokens,
        num_added_special_tokens=special_tokens,
        unk_token_id=getattr(tokenizer, "unk_token_id", None),
        residue_field=field_name,
    )
    LOGGER.info("Startup tokenization %s -> %s (special tokens: %d, residue positions: %d)",
                sequence, encoded.get("tokens"), special_tokens, len(positions))
    return RuntimeInspection(
        tokenization=report,
        output_type=type(out).__name__,
        output_attributes=[a for a in dir(out) if not a.startswith("_")][:40],
        hidden_state_shape=list(hidden.shape),
        residue_field=field_name,
        model_max_positions=max_positions,
        special_tokens=measured_special,
        residue_capacity=capacity,
        dtype=str(dtype).replace("torch.", ""),
        device=str(device),
    )


# ---------------------------------------------------------------------------
# Special-residue support (Section 6) -- full tokenizer -> model -> extraction path
# ---------------------------------------------------------------------------

@dataclass
class SpecialResidueSupport:
    symbol: str
    supported: bool
    primary_criterion_passed: bool
    diagnostic_unk: bool
    test_sequence: str
    residue_tokens: List[str]
    detail: str
    token_id: Optional[int] = None
    token_str: Optional[str] = None
    is_distinct_vocab_entry: bool = False
    maps_to_one_hidden_state: bool = False
    neighbor_alignment: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "supported": self.supported,
            "primary_criterion_passed": self.primary_criterion_passed,
            "diagnostic_token_id_is_not_unk": self.diagnostic_unk,
            "test_sequence": self.test_sequence,
            "residue_tokens": self.residue_tokens,
            "detail": self.detail,
            "token_id": self.token_id,
            "token_str": self.token_str,
            "is_distinct_vocab_entry": self.is_distinct_vocab_entry,
            "maps_to_one_hidden_state": self.maps_to_one_hidden_state,
            "neighbor_alignment": self.neighbor_alignment,
        }


def _run_forward(model: Any, tokenizer: Any, sequence: str, device: torch.device,
                 dtype: torch.dtype) -> Tuple[Dict[str, Any], List[int], Any]:
    encoded = encode_chunk(tokenizer, sequence)
    positions = derive_residue_positions(encoded, len(sequence), context=f"support({sequence!r})")
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    attn = encoded.get("attention_mask")
    attention = torch.tensor([attn], dtype=torch.long, device=device) if attn else None
    was_training = model.training
    model.eval()
    with torch.no_grad():
        try:
            out = model(input_ids=ids, attention_mask=attention, output_hidden_states=True)
        except TypeError:
            out = model(input_ids=ids, attention_mask=attention)
    if was_training:
        model.train()
    return encoded, positions, out


def _verify_support_alignment(encoded: Dict[str, Any], positions: Sequence[int], sequence: str,
                               symbol: str, context: str) -> List[str]:
    """Verify residue order for a special-residue support probe.

    Every position must map to exactly one residue, and the neighbours of the
    probe symbol must be unchanged. The symbol's own position may be occupied by
    the tokenizer's unk token (that is an *unsupported* symbol, reported by the
    diagnostic criterion) - what must never happen is a neighbour shifting,
    merging or disappearing.
    """
    tokens = encoded.get("tokens") or []
    symbol_index = sequence.index(symbol)
    derived: List[str] = []
    for residue_index, position in enumerate(positions):
        if position >= len(tokens):
            raise AlignmentError(
                f"{context}: cannot map token position {position} back to a residue")
        token = tokens[position].replace("▁", "").replace("Ġ", "").replace(" ", "")
        if residue_index == symbol_index and len(stripped_upper(token)) != 1:
            derived.append(symbol)  # occupied by an unknown token; flagged as diagnostic
            continue
        if len(stripped_upper(token)) != 1:
            raise AlignmentError(
                f"{context}: token {token!r} at residue position {residue_index} is not a single "
                f"residue; a neighbour of the probe symbol was merged or shifted."
            )
        derived.append(stripped_upper(token))
    if derived != list(sequence):
        raise AlignmentError(
            f"{context}: residue order mismatch. tokenizer={derived} vs sequence={list(sequence)}")
    return derived


def stripped_upper(token: str) -> str:
    return token.replace("▁", "").replace("Ġ", "").strip().upper()


def test_special_residue_support(model: Any, tokenizer: Any, symbol: str,
                                 device: Optional[torch.device] = None,
                                 dtype: torch.dtype = torch.float32) -> SpecialResidueSupport:
    """Execute the COMPLETE tokenizer -> model -> residue-extraction path for ``symbol``.

    PRIMARY criterion: exactly one residue-aligned hidden-state position, with
    the full residue order ``A, C, D, <sym>, E, F, G`` preserved (no neighbour
    shift/merge/disappearance). DIAGNOSTIC criterion: token id != unk id and token string matches symbol.
    """
    device = device or torch.device("cpu")
    test_sequence = f"ACD{symbol}EFG"
    encoded, positions, out = _run_forward(model, tokenizer, test_sequence, device, dtype)

    detail = ""
    primary_ok = False
    maps_to_one_hidden_state = False
    neighbor_alignment = False
    residue_tokens: List[str] = []
    try:
        if len(positions) != len(test_sequence):
            detail = (f"tokenizer produced {len(positions)} residue positions for "
                      f"{len(test_sequence)} residues")
        else:
            residue_tokens = _verify_support_alignment(encoded, positions, test_sequence, symbol,
                                                       context=f"support({symbol})")
            neighbor_alignment = True
            ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
            residues = extract_residue_hidden_states(out, ids, positions, len(test_sequence))
            if residues.shape[0] != len(test_sequence):
                detail = f"extracted {residues.shape[0]} residue vectors, expected {len(test_sequence)}"
            else:
                maps_to_one_hidden_state = True
                primary_ok = True
                detail = "one residue-aligned hidden state per residue; neighbours unshifted"
    except (AlignmentError, ValueError, KeyError) as exc:
        detail = f"{type(exc).__name__}: {exc}"

    encoded_ids = encoded.get("input_ids", [])
    tokens = encoded.get("tokens", [])
    unk_id = getattr(tokenizer, "unk_token_id", None)
    symbol_token_position = positions[test_sequence.index(symbol)] if (positions and test_sequence.index(symbol) < len(positions)) else None

    token_id: Optional[int] = None
    token_str: Optional[str] = None
    if symbol_token_position is not None and symbol_token_position < len(encoded_ids):
        token_id = int(encoded_ids[symbol_token_position])
    if symbol_token_position is not None and symbol_token_position < len(tokens):
        token_str = str(tokens[symbol_token_position])

    diagnostic_ok = False
    if token_id is not None and unk_id is not None:
        diagnostic_ok = token_id != int(unk_id)

    clean_token_str = stripped_upper(token_str) if token_str else None
    is_distinct_vocab_entry = bool(diagnostic_ok and clean_token_str == symbol.upper())

    if not diagnostic_ok:
        detail += (" | DIAGNOSTIC: the symbol's token id is the tokenizer unk id, so the symbol is "
                   "not natively supported (an id being returned is NOT support)")
    elif not is_distinct_vocab_entry:
        detail += f" | DIAGNOSTIC: token string {token_str!r} does not correspond to symbol {symbol!r}"

    supported = bool(primary_ok and diagnostic_ok and is_distinct_vocab_entry)

    return SpecialResidueSupport(
        symbol=symbol,
        supported=supported,
        primary_criterion_passed=primary_ok,
        diagnostic_unk=diagnostic_ok,
        test_sequence=test_sequence,
        residue_tokens=residue_tokens,
        detail=detail,
        token_id=token_id,
        token_str=token_str,
        is_distinct_vocab_entry=is_distinct_vocab_entry,
        maps_to_one_hidden_state=maps_to_one_hidden_state,
        neighbor_alignment=neighbor_alignment,
    )


def probe_special_residue_support(model: Any, tokenizer: Any,
                                  device: Optional[torch.device] = None,
                                  dtype: torch.dtype = torch.float32,
                                  symbols: str = SPECIAL_RESIDUES) -> Dict[str, SpecialResidueSupport]:
    results: Dict[str, SpecialResidueSupport] = {}
    for symbol in symbols:
        result = test_special_residue_support(model, tokenizer, symbol, device=device, dtype=dtype)
        results[symbol] = result
        LOGGER.info("Special-residue support %s: supported=%s (%s)", symbol, result.supported,
                    result.detail)
    return results


def support_map(results: Dict[str, SpecialResidueSupport]) -> Dict[str, bool]:
    return {symbol: result.supported for symbol, result in results.items()}


# ---------------------------------------------------------------------------
# OOM policy (Section 14)
# ---------------------------------------------------------------------------

@contextmanager
def _cuda_oom_guard(sequence_hash: Optional[str], sequence_length: Optional[int]):
    """Wrap one inference call. Never retries with a different dtype, chunk size
    or device: an OOM either propagates as fatal or, if the caller has verified
    a safe state, is downgraded to a single row-level failure."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - needs GPU
        LOGGER.error("CUDA OOM while encoding sequence_hash=%s length=%s: %s",
                     sequence_hash, sequence_length, exc)
        raise CudaOutOfMemoryError(
            f"CUDA out of memory for sequence_hash={sequence_hash} (length={sequence_length}): {exc}",
            sequence_hash=sequence_hash, sequence_length=sequence_length) from exc
    except RuntimeError as exc:  # pragma: no cover - some builds raise RuntimeError
        if "out of memory" not in str(exc).lower():
            raise
        LOGGER.error("CUDA OOM while encoding sequence_hash=%s length=%s: %s",
                     sequence_hash, sequence_length, exc)
        raise CudaOutOfMemoryError(
            f"CUDA out of memory for sequence_hash={sequence_hash} (length={sequence_length}): {exc}",
            sequence_hash=sequence_hash, sequence_length=sequence_length) from exc


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, CudaOutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class TargetProteinEncoder:
    """Reusable module: frozen ESMC + residue mean pooling (pooling point A).

    ``forward``/``encode`` returns a 1-D ``float32`` CPU tensor whose width is
    ``model.config.hidden_size`` (asserted == 1152 for the required ESMC-600M
    checkpoint). The optional 1152->256 projection is a **separate** module
    (:class:`esmc_target.projection.TargetProjection`) and is never fused here.
    """

    def __init__(self, resolved: ResolvedModel, device: torch.device,
                 dtype: str = "float32", chunk_size: Any = "auto",
                 overlap: int = 256,
                 model_max_positions_override: Optional[int] = None) -> None:
        self.resolved = resolved
        self.device = device
        self.dtype_name = dtype
        self.dtype = DTYPE_MAP[dtype]
        self.hidden_size = int(resolved.hidden_size)
        if self.hidden_size != REQUIRED_HIDDEN_SIZE:
            raise ModelResolutionError(
                f"hidden_size must be {REQUIRED_HIDDEN_SIZE} for ESMC-600M, got {self.hidden_size}")
        self.model = resolved.model
        self.tokenizer = resolved.tokenizer
        self.overlap = int(overlap)
        if model_max_positions_override is not None:
            if int(model_max_positions_override) > resolved.model_max_positions:
                raise CapacityError(
                    f"model_max_positions_override ({model_max_positions_override}) exceeds actual "
                    f"model position capacity ({resolved.model_max_positions})"
                )
        self.model_max_positions = int(model_max_positions_override or resolved.model_max_positions)
        self.special_tokens = derive_special_token_count(self.tokenizer)
        self.residue_capacity = derive_residue_capacity(self.model_max_positions, self.special_tokens)
        self.chunk_size = resolve_chunk_size(chunk_size, self.residue_capacity)
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be strictly smaller than chunk_size ({self.chunk_size})")
        self.chunk_forward_passes = 0
        self._freeze()

    # -- freezing ------------------------------------------------------
    def _freeze(self) -> None:
        """Frozen backbone: no grads, eval mode, inference dtype (Section 4)."""
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        if self.dtype is not torch.float32:
            self.model.to(dtype=self.dtype)
        self.model.to(device=self.device)

    def train(self, mode: bool = True):  # noqa: D401 - never allow train mode
        raise FatalError(
            "TargetProteinEncoder is frozen: this pipeline has zero trainable parameters and "
            "refuses to enter training mode."
        )

    def parameters(self):  # pragma: no cover - convenience
        return self.model.parameters()

    # -- inference -----------------------------------------------------
    def _forward_chunk(self, chunk_sequence: str, sequence_hash: str) -> torch.Tensor:
        """One ESMC forward pass -> residue-aligned ``[n_residues, hidden]`` states."""
        with _cuda_oom_guard(sequence_hash, len(chunk_sequence)):
            encoded, positions, out = _run_forward(self.model, self.tokenizer, chunk_sequence,
                                                  self.device, self.dtype)
            ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.device)
            residues = extract_residue_hidden_states(out, ids, positions, len(chunk_sequence))
            verify_residue_order(encoded, positions, chunk_sequence,
                                 context=f"encode(hash={sequence_hash[:12]})")
        self.chunk_forward_passes += 1
        total_tokens = len(encoded["input_ids"])
        if total_tokens > self.model_max_positions:  # pragma: no cover - guarded by planner
            raise AlignmentError(
                f"chunk of {len(chunk_sequence)} residues produced {total_tokens} tokens, exceeding "
                f"model_max_positions={self.model_max_positions}")
        return residues.detach().to(dtype=torch.float32, device="cpu")

    def encode(self, cleaned_sequence: str, force_chunk_size: Optional[int] = None) -> ProteinEncoding:
        """Encode one cleaned sequence into a single protein embedding."""
        seq_hash = sequence_hash(cleaned_sequence)
        length = len(cleaned_sequence)
        chunk_size = int(force_chunk_size) if force_chunk_size else self.chunk_size
        if force_chunk_size and force_chunk_size > self.residue_capacity:
            raise ValueError(
                f"forced chunk_size={force_chunk_size} exceeds residue capacity {self.residue_capacity}")
        intervals = plan_chunks(length, chunk_size, min(self.overlap, chunk_size - 1))
        accumulator = ResidueAccumulator(length=length, hidden_size=self.hidden_size)
        for start, end in intervals:
            chunk_residues = cleaned_sequence[start:end]
            states = self._forward_chunk(chunk_residues, seq_hash)
            accumulator.add(start, end, states)
            del states
        coverage = accumulator.assert_full_coverage()
        embedding = accumulator.pooled().to(dtype=torch.float32, device="cpu").contiguous()
        validate_embedding(embedding, self.hidden_size, sequence_hash=seq_hash)
        return ProteinEncoding(
            sequence=cleaned_sequence,
            sequence_hash=seq_hash,
            embedding=embedding,
            num_chunks=accumulator.num_chunks,
            chunk_boundaries=list(intervals),
            coverage=coverage,
            diagnostics=compute_diagnostics(embedding).to_dict(),
        )

    forward = encode

    def metadata(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "model_max_positions": self.model_max_positions,
            "special_tokens": self.special_tokens,
            "residue_capacity": self.residue_capacity,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "dtype": self.dtype_name,
            "device": str(self.device),
            "pooling_method": "residue_mean",
            "frozen": True,
        }
