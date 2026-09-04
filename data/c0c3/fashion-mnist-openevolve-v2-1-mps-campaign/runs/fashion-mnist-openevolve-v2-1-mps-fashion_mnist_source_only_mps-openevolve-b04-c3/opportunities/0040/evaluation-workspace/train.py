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


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(self.depthwise(inputs)))
        hidden = self.norm2(self.pointwise(hidden))
        return F.gelu(inputs + hidden)


class PartAttentionPool(nn.Module):
    def __init__(self, channels: int, parts: int = 4) -> None:
        super().__init__()
        self.attention = nn.Conv2d(channels, parts, kernel_size=1)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

        position = torch.zeros(1, parts, 7, 7)
        position[:, 0, :4, :4] = 1.0
        position[:, 1, :4, 3:] = 1.0
        position[:, 2, 3:, :4] = 1.0
        position[:, 3, 3:, 3:] = 1.0
        self.position_bias = nn.Parameter(position)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scores = self.attention(features) + self.position_bias
        weights = F.softmax(scores.flatten(2), dim=-1)
        values = features.flatten(2).transpose(1, 2)
        parts = torch.bmm(weights, values).flatten(1)
        global_summary = features.mean(dim=(2, 3))
        return torch.cat((global_summary, parts), dim=1)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            ResidualDepthwiseBlock(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualDepthwiseBlock(96),
        )
        self.pool = PartAttentionPool(96)
        self.classifier = nn.Sequential(
            nn.Linear(96 * 5, 372),
            nn.LayerNorm(372),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(372, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.stem(images)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = 0.5 * (logits + flipped_logits)
        return logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)


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
        images.flip(-1),
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
    del step, total_steps
    return F.cross_entropy(model(images), labels)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = step / max(total_steps, 1)
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
