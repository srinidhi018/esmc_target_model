"""Shared fixtures.

**The ESMC model is mocked for every unit test** - no download is required and
no test fabricates a scientific result. The mock reproduces the properties the
pipeline actually depends on: CLS/EOS wrapping, a special-token mask, a
``last_hidden_state`` of width 1152, ``max_position_embeddings``, per-residue
position-wise hidden states (so alignment/pooling bugs are detectable), and a
configurable vocabulary that can map an unknown symbol to ``unk_token_id``.

``tests/test_integration_esmc.py`` is the single opt-in test that really
downloads ESMC; it is skipped unless ``ESMC_REAL_RUN=1``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from esmc_target.config import AppConfig, load_config  # noqa: E402
from esmc_target.esmc_encoder import ResolvedModel  # noqa: E402
from esmc_target.utils import sha256_json  # noqa: E402

HIDDEN = 1152
MAX_POSITIONS = 2048
STANDARD = "ACDEFGHIKLMNPQRSTVWY"
SPECIALS = "XBZJUO"
VOCAB = ["<cls>", "<eos>", "<pad>", "<unk>"] + list(STANDARD + SPECIALS)
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
CLS_ID, EOS_ID, PAD_ID, UNK_ID = 0, 1, 2, 3


class MockOutput:
    """Mimics a HF model output object (``last_hidden_state`` + logits)."""

    def __init__(self, last_hidden_state: torch.Tensor, logits: Optional[torch.Tensor] = None) -> None:
        self.last_hidden_state = last_hidden_state
        self.logits = logits


class MockEsmcConfig:
    def __init__(self, hidden_size: int = HIDDEN, max_position_embeddings: int = MAX_POSITIONS,
                 architectures: Sequence[str] = ("EsmcForMaskedLM",),
                 unk_tokens: Sequence[str] = ()) -> None:
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.architectures = list(architectures)
        self.model_type = "esmc"
        self.num_hidden_layers = 33
        self._commit_hash = "mockcommit0000"
        # Tokens this checkpoint cannot encode; anything else is native.
        self.unk_tokens = set(unk_tokens)

    def to_dict(self) -> Dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "max_position_embeddings": self.max_position_embeddings,
            "architectures": self.architectures,
            "model_type": self.model_type,
            "num_hidden_layers": self.num_hidden_layers,
        }


class MockTokenizer:
    """CLS + residues + EOS, with a real special-token mask."""

    cls_token_id = CLS_ID
    eos_token_id = EOS_ID
    sep_token_id = None
    bos_token_id = None
    pad_token_id = PAD_ID

    def __init__(self, unk_tokens: Sequence[str] = ()) -> None:
        self.unk_tokens = set(unk_tokens)

    @property
    def unk_token_id(self) -> int:
        return UNK_ID

    def _encode_residue(self, symbol: str) -> int:
        if symbol in self.unk_tokens:
            return UNK_ID
        return TOKEN_TO_ID.get(symbol, UNK_ID)

    def __call__(self, text: str, return_tensors: Optional[str] = None,
                 add_special_tokens: bool = True, **kwargs) -> Dict[str, object]:
        ids = [CLS_ID]
        mask = [1]
        for symbol in str(text):
            ids.append(self._encode_residue(symbol))
            mask.append(0)
        if add_special_tokens:
            ids.append(EOS_ID)
            mask.append(1)
        out: Dict[str, object] = {"input_ids": ids, "special_tokens_mask": mask,
                                  "attention_mask": [1] * len(ids)}
        if return_tensors == "pt":
            out = {k: torch.tensor([v]) for k, v in out.items()}
        return out

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        return [VOCAB[int(i)] if int(i) < len(VOCAB) else "<unk>" for i in ids]

    def to_dict(self) -> Dict[str, object]:
        return {"unk_tokens": sorted(self.unk_tokens), "cls": "<cls>", "eos": "<eos>"}


class MockEsmcModel(torch.nn.Module):
    """Position-wise deterministic encoder: residue i gets a vector derived
    from its token id, so residue alignment and pooling are testable."""

    def __init__(self, config: Optional[MockEsmcConfig] = None) -> None:
        super().__init__()
        self.config = config or MockEsmcConfig()
        self.forward_calls = 0
        self.max_positions_seen = 0
        self.scale = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> MockOutput:
        self.forward_calls += 1
        n = int(input_ids.shape[-1])
        self.max_positions_seen = max(self.max_positions_seen, n)
        if n > int(self.config.max_position_embeddings):
            raise RuntimeError(
                f"input length {n} exceeds the model's max positions "
                f"{self.config.max_position_embeddings}")
        states = torch.zeros((n, int(self.config.hidden_size)), dtype=torch.float32)
        for position in range(n):
            token = int(input_ids[0, position])
            # Distinct, finite, non-zero per-token signal.
            base = float(token + 1)
            states[position] = torch.arange(
                int(self.config.hidden_size), dtype=torch.float32) * 0.01 + base
        return MockOutput(last_hidden_state=states.unsqueeze(0))


def make_resolved(hidden_size: int = HIDDEN, max_position_embeddings: int = MAX_POSITIONS,
                  unk_tokens: Sequence[str] = ()) -> ResolvedModel:
    config = MockEsmcConfig(hidden_size=hidden_size, max_position_embeddings=max_position_embeddings)
    tokenizer = MockTokenizer(unk_tokens=unk_tokens)
    model = MockEsmcModel(config)
    return ResolvedModel(
        model_id="biohub/ESMC-600M-hf", model_revision="mockcommit0000",
        tokenizer_revision="mockcommit0000", model=model, tokenizer=tokenizer,
        model_class="MockEsmcModel", tokenizer_class="MockTokenizer",
        architecture="EsmcForMaskedLM", hidden_size=hidden_size,
        model_max_positions=max_position_embeddings, model_config=config,
        model_config_hash=sha256_json(config.to_dict()),
        tokenizer_config_hash=sha256_json(tokenizer.to_dict()),
        transformers_version="mock", api_used="mock", owner="biohub",
        applies_final_layer_norm=False, effective_attention_implementation="eager",
        requested_candidates=["biohub/ESMC-600M-hf"])


@pytest.fixture
def resolved_model() -> ResolvedModel:
    return make_resolved()


@pytest.fixture
def resolved_model_no_u() -> ResolvedModel:
    """Mock checkpoint whose tokenizer cannot encode U (maps to unk)."""
    return make_resolved(unk_tokens=("U",))


@pytest.fixture
def encoder(resolved_model: ResolvedModel):
    from esmc_target.esmc_encoder import TargetProteinEncoder
    return TargetProteinEncoder(resolved=resolved_model, device=torch.device("cpu"),
                                dtype="float32", chunk_size="auto", overlap=256)


@pytest.fixture
def config() -> AppConfig:
    cfg = load_config(None)
    cfg.output.diagnostics = False
    return cfg


def protein(length: int, seed: int = 0) -> str:
    """Deterministic pseudo-protein of a given length (no RNG at run time)."""
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(alphabet[(i * 7 + seed) % 20] for i in range(length))
