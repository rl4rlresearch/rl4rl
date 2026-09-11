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
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                24, 48, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.stage2_shortcut = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(48),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(
                96,
                96,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(96),
        )
        self.stage3_shortcut = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96 * 6, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(
            self.stage2(features) + self.stage2_shortcut(features)
        )
        features = F.gelu(
            self.stage3(features) + self.stage3_shortcut(features)
        )
        pooled = torch.cat(
            (
                F.adaptive_avg_pool2d(features, 1).flatten(1),
                F.adaptive_avg_pool2d(features, 2).flatten(1),
                F.adaptive_max_pool2d(features, 1).flatten(1),
            ),
            dim=1,
        )
        return self.classifier(pooled)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)
        padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
        logits_sum = None
        for offset_y in range(3):
            for offset_x in range(3):
                view = padded[
                    :,
                    :,
                    offset_y : offset_y + 28,
                    offset_x : offset_x + 28,
                ]
                views = torch.cat((view, view.flip(-1)), dim=0)
                logits = self._forward_once(views)
                original_logits, flipped_logits = logits.chunk(2, dim=0)
                view_logits = original_logits + flipped_logits
                if logits_sum is None:
                    logits_sum = view_logits
                else:
                    logits_sum = logits_sum + view_logits
        return 1.2 * logits_sum / 18.0


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    offsets = torch.randint(0, 5, (2,))
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    images = padded[:, :, offset_y : offset_y + 28, offset_x : offset_x + 28]
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
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.1 + 0.9 * progress / warmup_fraction
    else:
        cosine_progress = (
            progress - warmup_fraction
        ) / (1.0 - warmup_fraction)
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
