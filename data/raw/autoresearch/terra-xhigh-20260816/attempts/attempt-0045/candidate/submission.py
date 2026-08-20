"""AdderBoard submission wrapper for this Autoresearch workspace.

The editable model and data implementation live in ``src/``.  This file is
part of the candidate: it must continue to expose the official AdderBoard
``build_model`` and generic autoregressive ``add`` interface.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch


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
            f"Missing {checkpoint}. Train the candidate before official verification."
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
        position_rank=config.get("position_rank"),
        tie_norms_across_layers=config.get("tie_norms_across_layers", False),
        no_norm_bias=config.get("no_norm_bias", False),
        factorized_positions=config.get("factorized_positions", False),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def build_model():
    """Return the trained candidate and honest, deduplicated metadata."""
    model, config = _build_from_checkpoint()
    unique_parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "name": "RL4RL Autoresearch candidate",
        "author": "configured in run manifest",
        "params": unique_parameters,
        "architecture": (
            "decoder-only addition transformer "
            f"d={config['d_model']}, h={config['n_heads']}, "
            f"L={config['n_layers']}, ff={config['ff_dim']}, "
            f"position_rank={config.get('position_rank', 'full')}"
            f", tie_norms_across_layers={config.get('tie_norms_across_layers', False)}"
            f", no_norm_bias={config.get('no_norm_bias', False)}"
            f", factorized_positions={config.get('factorized_positions', False)}"
        ),
        "tricks": ["autoregressive greedy decoding", "weights from checkpoint"],
    }
    return model, metadata


@torch.no_grad()
def add(model, a: int, b: int) -> int:
    """Generic greedy autoregressive decoding; no addition logic lives here."""
    input_ids = [BOS_ID] + encode(preprocess(a, b))
    tokens = torch.tensor([input_ids], dtype=torch.long)
    generated = model.generate(tokens, max_new_tokens=OUT_DIGITS + 1, eos_id=EOS_ID)
    output_ids = generated[0, len(input_ids) :].tolist()
    if EOS_ID in output_ids:
        output_ids = output_ids[: output_ids.index(EOS_ID)]
    raw_output = "".join(ID_TO_TOKEN.get(token, "?") for token in output_ids)
    return postprocess(raw_output)
