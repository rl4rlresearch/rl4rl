"""Editable image-classification research program.

The verification harness owns the data split, example budget, evaluation, and
timing.  This file owns the learned model and the trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.skip = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.conv5 = nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(96)
        self.pool = nn.MaxPool2d(2)
        self.spatial_pool = nn.AdaptiveAvgPool2d(3)
        self.fc1 = nn.Linear(96 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(images)), inplace=True)
        x = F.relu(x + self.bn2(self.conv2(x)), inplace=True)
        x = self.pool(x)

        residual = self.skip(x)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(residual + self.bn4(self.conv4(x)), inplace=True)
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)), inplace=True)
        x = self.spatial_pool(x).flatten(1)
        x = F.dropout(x, p=0.15, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, p=0.10, training=self.training)
        return self.fc2(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits
        flipped_logits = self._forward_once(images.flip(-1))
        return torch.logaddexp(
            F.log_softmax(logits, dim=1),
            F.log_softmax(flipped_logits, dim=1),
        ) - math.log(2.0)


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay = [parameter for parameter in model.parameters() if parameter.ndim > 1]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim <= 1]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=6e-4,
        betas=(0.9, 0.99),
    )


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    flip_mask = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels


def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    completed = step + 1
    warmup_steps = max(1, total_steps // 20)
    if completed <= warmup_steps:
        multiplier = 0.2 + 0.8 * completed / warmup_steps
    else:
        progress = min(
            1.0,
            (completed - warmup_steps)
            / max(total_steps - warmup_steps, 1),
        )
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
