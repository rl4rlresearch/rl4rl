"""Fixed interface for a trained pair-token addition transformer.

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
