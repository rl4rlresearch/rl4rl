"""A 1,012-parameter learned transformer for exact four-digit addition.

Editable model, representation, optimizer, loss, and training-step budget.
Verification owns disjoint data, fresh initialization, decoding, and scoring.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 2048
GRAD_CLIP_NORM = 1.0

# Editable update count, not an evaluator-imposed limit.
TRAINING_STEPS = 400
# The schedule horizon is independent of the number of updates.
LR_SCHEDULE_STEPS = 1000
OUTPUT_TOKENS = 6


class CausalSelfAttention(nn.Module):
    """Learned relative-position attention with content-dependent values.

    Each head learns a score for every causal distance. No offset is selected
    in advance. This is a position-based attention variant, with independent
    learned value and output projections.
    """

    def __init__(self, width: int, heads: int, sequence_length: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.relative_bias = nn.Embedding(sequence_length, heads)
        self.value = nn.Linear(width, width)
        self.projection = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        positions = torch.arange(length, device=hidden.device)
        distances = positions[:, None] - positions[None, :]
        scores = self.relative_bias(distances.clamp_min(0)).permute(2, 0, 1)
        scores = scores.masked_fill(distances[None] < 0, float("-inf"))
        probabilities = scores.softmax(dim=-1)
        value = (
            self.value(hidden)
            .view(batch, length, self.heads, self.head_width)
            .transpose(1, 2)
        )
        mixed = probabilities.unsqueeze(0) @ value
        return self.projection(
            mixed.transpose(1, 2).contiguous().view(batch, length, width)
        )


class TransformerBlock(nn.Module):
    def __init__(
        self, width: int, heads: int, feedforward_width: int, sequence_length: int
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads, sequence_length)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.GELU(),
            nn.Linear(feedforward_width, width),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class TinyAdditionTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int = 114,
        sequence_length: int = 11,
        width: int = 6,
        heads: int = 2,
        feedforward_width: int = 6,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.token_embedding = nn.Sequential(
            nn.Embedding(vocabulary_size, 3), nn.Linear(3, width, bias=False)
        )
        self.position_embedding = nn.Embedding(sequence_length, width)
        self.block = TransformerBlock(width, heads, feedforward_width, sequence_length)
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Sequential(
            nn.Linear(width, 3, bias=False), nn.Linear(3, vocabulary_size, bias=False)
        )
        self.apply(self._initialize_weights)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        length = token_ids.shape[1]
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.block(hidden)
        return self.output(self.final_norm(hidden))


def build_model() -> nn.Module:
    return TinyAdditionTransformer()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    routing = [p for n, p in model.named_parameters() if "relative_bias" in n]
    other = [p for n, p in model.named_parameters() if "relative_bias" not in n]
    return torch.optim.AdamW(
        [
            {"params": routing, "lr": 0.1, "initial_lr": 0.1},
            {"params": other, "lr": 0.02, "initial_lr": 0.02},
        ],
        weight_decay=0.01,
    )


def training_loss(
    model: nn.Module,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    logits = model(token_ids)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


def after_optimizer_step(
    optimizer: torch.optim.Optimizer, step: int, total_steps: int
) -> None:
    progress = min(step / max(LR_SCHEDULE_STEPS, 1), 1.0)
    multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * multiplier


def encode_inputs(
    left_digits: torch.Tensor, right_digits: torch.Tensor
) -> torch.Tensor:
    """Editable reversible operand formatting; each input is [batch, 4], LSB first.

    Pair tokens are the seed representation only. Never compute answers here.
    """
    pairs = 10 + left_digits * 10 + right_digits
    return torch.cat(
        (torch.full_like(pairs[:, :1], 111), pairs, torch.full_like(pairs[:, :1], 110)),
        dim=1,
    )


def encode_targets(answer_digits: torch.Tensor) -> torch.Tensor:
    """Format supplied labels; inference never passes labels to this function."""
    return torch.cat((answer_digits, torch.full_like(answer_digits[:, :1], 112)), dim=1)


def decode_targets(generated: torch.Tensor) -> torch.Tensor:
    """Formatting only: return five predicted decimal digits, LSB first.

    Receives generated answer tokens only, never operands or expected answers.
    """
    return generated[:, :5]
