"""Editable image-classification research program.

The verification harness owns the data split, example budget, evaluation, and
timing.  This file owns the learned model and the trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class TokenBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            num_heads=4,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.transition = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.MaxPool2d(2)
        token_width = 96
        self.token_projection = nn.Conv2d(
            64, token_width, kernel_size=1, bias=False
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, token_width))
        self.position_embedding = nn.Parameter(
            torch.empty(1, 50, token_width)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.token_blocks = nn.ModuleList(
            TokenBlock(token_width) for _ in range(2)
        )
        self.final_norm = nn.LayerNorm(token_width)
        self.token_dropout = nn.Dropout(0.15)
        self.head = nn.Linear(token_width, 10)

    def _forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = self.pool(features)
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        return self.pool(features)

    def _forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        features = self.token_projection(self._forward_features(images))
        tokens = features.flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = tokens + self.position_embedding
        for block in self.token_blocks:
            tokens = block(tokens)
        representation = self.final_norm(tokens[:, 0])
        return self.head(self.token_dropout(representation))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_logits(images)
        if self.training:
            return logits
        return 1.2112 * logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.1e-3,
        betas=(0.9, 0.99),
        weight_decay=2e-4,
    )
    optimizer.ema_model = model
    optimizer.ema_state = None
    optimizer.ema_parameter_names = {
        name for name, _ in model.named_parameters()
    }
    return optimizer


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels


def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = min(step / max(total_steps, 1), 1.0)
    dropout_decay = max(progress - 0.50, 0.0) / 0.50
    model.token_dropout.p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    decay_progress = max(progress - 0.10, 0.0) / 0.90
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
    for group in optimizer.param_groups:
        group["lr"] = 2.1e-3 * multiplier

    if progress >= 0.50:
        current_state = optimizer.ema_model.state_dict()
        with torch.no_grad():
            if optimizer.ema_state is None:
                optimizer.ema_state = {
                    name: value.detach().clone()
                    for name, value in current_state.items()
                }
            else:
                for name, value in current_state.items():
                    averaged = optimizer.ema_state[name]
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)

            if step >= total_steps:
                for name, value in current_state.items():
                    value.copy_(optimizer.ema_state[name])
