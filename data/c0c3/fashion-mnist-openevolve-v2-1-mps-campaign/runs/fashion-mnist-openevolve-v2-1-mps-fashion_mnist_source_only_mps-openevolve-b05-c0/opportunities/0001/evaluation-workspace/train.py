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
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.10),
            nn.Linear(96 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 10),
        )

    def _logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._logits(images)
        if self.training:
            return logits
        return 0.5 * (logits + self._logits(images.flip(-1)))


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=4.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if step < int(0.85 * total_steps):
        count = images.size(0)
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        top = torch.randint(0, 5, (count,), device=images.device)
        left = torch.randint(0, 5, (count,), device=images.device)
        batch_index = torch.arange(count, device=images.device)[:, None, None]
        row_index = top[:, None, None] + torch.arange(
            28, device=images.device
        )[None, :, None]
        column_index = left[:, None, None] + torch.arange(
            28, device=images.device
        )[None, None, :]
        images = padded[
            batch_index, 0, row_index, column_index
        ].unsqueeze(1)
        flip_mask = torch.rand(
            (count, 1, 1, 1), device=images.device
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
    progress = min(step / max(total_steps, 1), 1.0)
    warmup = 0.08
    start_lr = 4.0e-4
    peak_lr = 3.0e-3
    end_lr = 1.5e-4
    if progress < warmup:
        lr = start_lr + (peak_lr - start_lr) * progress / warmup
    else:
        decay_progress = (progress - warmup) / (1.0 - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        lr = end_lr + (peak_lr - end_lr) * cosine
    for group in optimizer.param_groups:
        group["lr"] = lr
