"""Subject-neutral support for the learned 10-digit addition task."""

from __future__ import annotations

NEUTRAL_TASK_ADAPTER = "ten_digit_addition_transformer_v1"
PAIR_TOKEN_TASK_ADAPTER = "ten_digit_addition_pair_transformer_v1"
PAIR_TOKEN_TASK_ADAPTER_V2 = "ten_digit_addition_pair_transformer_v2"
PAIR_TOKEN_TASK_ADAPTER_V3 = "ten_digit_addition_pair_transformer_v3"
NEUTRAL_PROMPT_PROFILE = "trained_transformer_optimizer_v1_5"
OPENEVOLVE_V2_PROMPT_PROFILE = "trained_transformer_openevolve_v2"
AUTORESEARCH_V17_PROMPT_PROFILE = "trained_transformer_optimizer_v1_7"
OPENEVOLVE_V21_PROMPT_PROFILE = "trained_transformer_openevolve_v2_1"
SUBJECT_NEUTRAL_PROTOCOL_VERSIONS = frozenset(
    {"1.5", "1.6", "1.7", "2.0", "2.1"}
)
SUBJECT_NEUTRAL_PROMPT_PROFILES = frozenset(
    {
        NEUTRAL_PROMPT_PROFILE,
        OPENEVOLVE_V2_PROMPT_PROFILE,
        AUTORESEARCH_V17_PROMPT_PROFILE,
        OPENEVOLVE_V21_PROMPT_PROFILE,
    }
)
SUBJECT_NEUTRAL_TASK_ADAPTERS = frozenset(
    {
        NEUTRAL_TASK_ADAPTER,
        PAIR_TOKEN_TASK_ADAPTER,
        PAIR_TOKEN_TASK_ADAPTER_V2,
        PAIR_TOKEN_TASK_ADAPTER_V3,
    }
)
ARTIFACT_CLEAN_PROTOCOL_VERSIONS = frozenset({"1.7", "2.1"})
ARTIFACT_CLEAN_PROMPT_PROFILES = frozenset(
    {AUTORESEARCH_V17_PROMPT_PROFILE, OPENEVOLVE_V21_PROMPT_PROFILE}
)
ARTIFACT_CLEAN_ASSUMPTION_PROMPT_PATHS = {
    AUTORESEARCH_V17_PROMPT_PROFILE: (
        "transformer_optimizer_v1_7/assumption_changing.md"
    ),
    OPENEVOLVE_V21_PROMPT_PROFILE: (
        "transformer_optimizer_openevolve_v2_1/assumption_changing.md"
    ),
}
OPERATOR_PROMPT_ROOT_ENV = "RL4RL_C0C3_OPERATOR_PROMPT_ROOT"

SANITIZED_SEED_PATHS = (
    "src/model.py",
    "src/data.py",
    "src/train.py",
    "checkpoints/best.pt",
)

PAIR_TOKEN_SANITIZED_SEED_PATHS = (
    "src/__init__.py",
    "src/model.py",
    "src/data.py",
    "src/eval.py",
    "src/train.py",
    "checkpoints/best.pt",
)

# Protocols 1.7 and 2.1 provide source and verified public results, never a
# pretrained checkpoint. Calibration and every candidate evaluation train in a
# separate evaluator workspace from a fresh initialization.
PAIR_TOKEN_SOURCE_ONLY_SEED_PATHS = tuple(
    path for path in PAIR_TOKEN_SANITIZED_SEED_PATHS if path != "checkpoints/best.pt"
)


def validate_v15_pairing(
    *,
    protocol_version: str,
    task_adapter: str,
    prompt_profile: str | None = None,
) -> None:
    """Fail closed if subject-neutral components are mixed with older strata."""

    is_subject_neutral = protocol_version in SUBJECT_NEUTRAL_PROTOCOL_VERSIONS
    expected_adapter = {
        "1.7": PAIR_TOKEN_TASK_ADAPTER_V3,
        "2.0": PAIR_TOKEN_TASK_ADAPTER_V2,
        "2.1": PAIR_TOKEN_TASK_ADAPTER_V3,
    }.get(protocol_version)
    if expected_adapter is not None and task_adapter != expected_adapter:
        raise ValueError(
            f"protocol {protocol_version} requires task adapter {expected_adapter}"
        )
    if is_subject_neutral and task_adapter not in SUBJECT_NEUTRAL_TASK_ADAPTERS:
        raise ValueError(
            "subject-neutral protocols require the subject-neutral task adapter"
        )
    if not is_subject_neutral and task_adapter in SUBJECT_NEUTRAL_TASK_ADAPTERS:
        raise ValueError(
            "the subject-neutral task adapter requires a subject-neutral protocol"
        )
    if prompt_profile is None:
        return
    expected_profile = {
        "1.7": AUTORESEARCH_V17_PROMPT_PROFILE,
        "2.0": OPENEVOLVE_V2_PROMPT_PROFILE,
        "2.1": OPENEVOLVE_V21_PROMPT_PROFILE,
    }.get(protocol_version, NEUTRAL_PROMPT_PROFILE)
    if is_subject_neutral and prompt_profile != expected_profile:
        raise ValueError(
            "subject-neutral prompt profile mismatch: "
            f"protocol {protocol_version} requires {expected_profile}"
        )
    if not is_subject_neutral and prompt_profile in SUBJECT_NEUTRAL_PROMPT_PROFILES:
        raise ValueError(
            "the subject-neutral prompt profile requires a subject-neutral protocol"
        )


NEUTRAL_SUBMISSION_WRAPPER = '''"""Fixed interface for the trained addition transformer.

The editable implementation lives in ``src/``. Verification always loads the
fresh checkpoint produced by ``src/train.py`` and uses generic autoregressive
decoding. This file is intentionally not editable.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    FIXED_SEQ_LEN,
    ID_TO_TOKEN,
    OUT_DIGITS,
    VOCAB_SIZE,
    encode,
    postprocess,
    preprocess,
)
from model import AdditionTransformer  # noqa: E402


def _checkpoint_path() -> Path:
    return ROOT / "checkpoints" / "best.pt"


def _build_from_checkpoint() -> tuple[AdditionTransformer, dict]:
    checkpoint = _checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing {checkpoint}. Train the implementation before verification."
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = AdditionTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ff_dim=config["ff_dim"],
        max_seq_len=FIXED_SEQ_LEN,
        dropout=0.0,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def _validate_learned_transformer(model: nn.Module) -> int:
    parameters = tuple(model.parameters())
    unique_parameters = sum(parameter.numel() for parameter in parameters)
    if unique_parameters <= 0:
        raise RuntimeError("The submitted model must have nonzero learned parameters.")
    if not any(
        parameter.requires_grad and parameter.numel() > 0
        for parameter in parameters
    ):
        raise RuntimeError("The submitted model has no nonempty trainable parameters.")

    attention_modules = [
        module
        for module in model.modules()
        if isinstance(module, nn.MultiheadAttention)
        or "attention" in module.__class__.__name__.lower()
    ]
    if not any(
        sum(parameter.numel() for parameter in module.parameters()) > 0
        for module in attention_modules
    ):
        raise RuntimeError(
            "The submitted model must contain a learned self-attention module."
        )
    return unique_parameters


def build_model():
    """Return the freshly trained model and deduplicated parameter metadata."""
    model, config = _build_from_checkpoint()
    unique_parameters = _validate_learned_transformer(model)
    metadata = {
        "name": "10-digit addition transformer",
        "author": "autonomous optimization run",
        "params": unique_parameters,
        "architecture": (
            "trained autoregressive transformer "
            f"d={config['d_model']}, h={config['n_heads']}, "
            f"L={config['n_layers']}, ff={config['ff_dim']}"
        ),
        "tricks": ["generic greedy decoding", "freshly trained checkpoint"],
    }
    return model, metadata


@torch.no_grad()
def add(model, a: int, b: int) -> int:
    """Decode only through model logits; no arithmetic solution is implemented here."""
    input_ids = [BOS_ID] + encode(preprocess(a, b))
    tokens = torch.tensor([input_ids], dtype=torch.long)
    generated = model.generate(tokens, max_new_tokens=OUT_DIGITS + 1, eos_id=EOS_ID)
    output_ids = generated[0, len(input_ids) :].tolist()
    if EOS_ID in output_ids:
        output_ids = output_ids[: output_ids.index(EOS_ID)]
    raw_output = "".join(ID_TO_TOKEN.get(token, "?") for token in output_ids)
    return postprocess(raw_output)
'''


PAIR_TOKEN_SUBMISSION_WRAPPER = (
    '''"""Fixed interface for a trained pair-token addition transformer.

The editable implementation lives in ``src/``. Verification always loads the
fresh checkpoint produced by ``src.train`` and uses generic autoregressive
decoding. This file is intentionally not editable.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import BOS_ID, preprocess, postprocess  # noqa: E402
from src.model import ModelConfig, TinyDecoderLM  # noqa: E402


def encode(tokens):
    """The task tokenizer already returns token IDs."""
    return list(tokens)


def _checkpoint_path() -> Path:
    return ROOT / "checkpoints" / "best.pt"


def _build_from_checkpoint() -> tuple[TinyDecoderLM, dict]:
    checkpoint = _checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing {checkpoint}. Train the implementation before verification."
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = TinyDecoderLM(ModelConfig(**config))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def _validate_learned_transformer(model: nn.Module) -> int:
    parameters = tuple(model.parameters())
    unique_parameters = sum(parameter.numel() for parameter in parameters)
    if unique_parameters <= 0:
        raise RuntimeError("The submitted model must have nonzero learned parameters.")
    if not any(
        parameter.requires_grad and parameter.numel() > 0
        for parameter in parameters
    ):
        raise RuntimeError("The submitted model has no nonempty trainable parameters.")
    if not any(
        sum(parameter.numel() for parameter in module.parameters()) > 0
        for module in model.modules()
        if "attention" in module.__class__.__name__.lower()
    ):
        raise RuntimeError(
            "The submitted model must contain a learned self-attention module."
        )
    return unique_parameters


def build_model():
    """Return the freshly trained model and deduplicated parameter metadata."""
    model, config = _build_from_checkpoint()
    unique_parameters = _validate_learned_transformer(model)
    metadata = {
        "name": "10-digit addition transformer",
        "author": "autonomous optimization run",
        "params": unique_parameters,
        "architecture": (
            "trained autoregressive transformer with pair-column input tokens "
            f"d={config['d_model']}, h={config['n_head']}, "
            f"L={config['n_layer']}, ff={config['d_ff']}"
        ),
        "tricks": ["generic greedy decoding", "freshly trained checkpoint"],
    }
    return model, metadata


@torch.no_grad()
def add(model, a: int, b: int) -> int:
    """Decode only through model logits; no arithmetic solution is implemented here."""
    input_ids = preprocess(a, b)
    tokens = torch.tensor([input_ids], dtype=torch.long)
    generated = model.generate(tokens, max_new_tokens=12)
    output_ids = generated[0, len(input_ids) :].tolist()
    return postprocess(output_ids)
'''
)
