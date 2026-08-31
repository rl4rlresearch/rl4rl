"""Editable learned transformer for exact four-digit addition.

The protected evaluator owns data generation, disjoint splits, training steps,
fresh initialization, decoding, accuracy measurement, and parameter counting.
This file owns only the learned model and trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 2048
GRAD_CLIP_NORM = 1.0

# The evaluator accepts a bounded literal policy, always appends the common
# terminal step, and stops the single uninterrupted trajectory at the first
# rung that reaches 99% exact accuracy.
EVALUATION_LADDER = [200, 400, 600, 1_000]


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, sequence_length: int) -> None:
        super().__init__()
        del sequence_length
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = query.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        value = value.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, width)
        return self.projection(mixed)


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
        width: int = 48,
        heads: int = 4,
        feedforward_width: int = 64,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.token_embedding = nn.Embedding(vocabulary_size, width)
        self.position_embedding = nn.Embedding(sequence_length, width)
        self.block = TransformerBlock(
            width, heads, feedforward_width, sequence_length
        )
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocabulary_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
    return torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)


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
    progress = step / max(total_steps, 1)
    multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 5e-3 * multiplier
