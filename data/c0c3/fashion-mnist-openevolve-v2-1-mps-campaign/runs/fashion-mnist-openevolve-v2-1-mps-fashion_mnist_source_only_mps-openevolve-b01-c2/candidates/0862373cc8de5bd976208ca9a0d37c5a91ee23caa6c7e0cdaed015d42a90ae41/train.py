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
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 0.5 * (logits + flipped_logits)
        return logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    augmented = images.clone()
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    augmented[flip_mask] = torch.flip(augmented[flip_mask], dims=(-1,))

    padding = 2
    padded = F.pad(
        augmented,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    crop_windows = padded.unfold(2, images.shape[-2], 1).unfold(
        3, images.shape[-1], 1
    )
    offset_y = torch.randint(
        0, 2 * padding + 1, (images.shape[0],), device=images.device
    )
    offset_x = torch.randint(
        0, 2 * padding + 1, (images.shape[0],), device=images.device
    )
    batch_indices = torch.arange(images.shape[0], device=images.device)
    augmented = crop_windows[
        batch_indices, :, offset_y, offset_x
    ].contiguous()
    return augmented, labels


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
    peak_lr = 3.0e-3
    warmup_steps = max(1, int(0.05 * total_steps))
    if step < warmup_steps:
        multiplier = 0.1 + 0.9 * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        multiplier = 0.02 + 0.98 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = peak_lr * multiplier
