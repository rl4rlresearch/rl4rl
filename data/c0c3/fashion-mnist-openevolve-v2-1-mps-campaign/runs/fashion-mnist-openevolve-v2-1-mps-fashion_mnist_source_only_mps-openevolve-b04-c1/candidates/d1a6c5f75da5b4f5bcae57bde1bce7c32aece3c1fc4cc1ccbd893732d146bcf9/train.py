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
        self.down1 = nn.Sequential(
            nn.Conv2d(
                32, 48, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(48),
        )
        self.down1_skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(
                48, 72, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(72),
        )
        self.down2_skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 72, kernel_size=1, bias=False),
            nn.BatchNorm2d(72),
        )
        self.residual3 = nn.Sequential(
            nn.Conv2d(72, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72 * 7 * 7, 32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 10),
        )

    def _forward_view(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = F.gelu(
            self.down1(features) + self.down1_skip(features)
        )
        features = F.gelu(features + self.residual2(features))
        features = F.gelu(
            self.down2(features) + self.down2_skip(features)
        )
        features = F.gelu(features + self.residual3(features))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_view(images)
        if self.training:
            return logits
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)


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
    del step, total_steps
    batch_size = images.shape[0]
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_logits = model(paired_images)
    logits = 0.5 * (
        paired_logits[:batch_size] + paired_logits[batch_size:]
    )
    return F.cross_entropy(logits, labels, label_smoothing=0.02)


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
                    if torch.is_floating_point(averaged):
                        averaged.lerp_(value.detach(), 0.01)
                    else:
                        averaged.copy_(value)

            if step >= total_steps:
                for name, value in current_state.items():
                    value.copy_(optimizer.ema_state[name])
