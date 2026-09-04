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
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "detail_kernels",
            torch.tensor(
                [
                    [
                        [
                            [-0.125, 0.0, 0.125],
                            [-0.250, 0.0, 0.250],
                            [-0.125, 0.0, 0.125],
                        ]
                    ],
                    [
                        [
                            [-0.125, -0.250, -0.125],
                            [0.0, 0.0, 0.0],
                            [0.125, 0.250, 0.125],
                        ]
                    ],
                    [
                        [
                            [0.0, 0.250, 0.0],
                            [0.250, -1.0, 0.250],
                            [0.0, 0.250, 0.0],
                        ]
                    ],
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        details = F.conv2d(padded, self.detail_kernels)
        represented = torch.cat((images, details), dim=1)
        return self.classifier(self.features(represented))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
        logits = []
        for view, weight in zip(views, view_weights):
            logits.append(self._forward_once(view) * weight)
            logits.append(self._forward_once(view.flip(-1)) * weight)
        return torch.stack(logits, dim=0).sum(dim=0) / (
            2.0 * sum(view_weights)
        )


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay = [parameter for parameter in model.parameters() if parameter.ndim > 1]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 4e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1.25e-4,
        betas=(0.9, 0.99),
    )
    optimizer._ema_shadow = [
        torch.zeros_like(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer._ema_updates = 0
    return optimizer


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    batch_indices = torch.arange(images.shape[0], device=images.device)
    offset_y = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    offset_x = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    patches = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    images = patches[batch_indices, :, offset_y, offset_x]
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
    return F.cross_entropy(model(images), labels, label_smoothing=0.015)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    ema_decay = 0.99
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    with torch.no_grad():
        optimizer._ema_updates += 1
        for shadow, parameter in zip(optimizer._ema_shadow, parameters):
            shadow.mul_(ema_decay).add_(parameter, alpha=1.0 - ema_decay)
        if step + 1 >= total_steps:
            correction = 1.0 - ema_decay ** optimizer._ema_updates
            for shadow, parameter in zip(optimizer._ema_shadow, parameters):
                parameter.lerp_(shadow / correction, 0.28125)

    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.06
    if progress < warmup_fraction:
        multiplier = 0.1 + 0.9 * progress / warmup_fraction
    else:
        cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        multiplier = 0.08 + 0.92 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 1.25e-3 * multiplier
