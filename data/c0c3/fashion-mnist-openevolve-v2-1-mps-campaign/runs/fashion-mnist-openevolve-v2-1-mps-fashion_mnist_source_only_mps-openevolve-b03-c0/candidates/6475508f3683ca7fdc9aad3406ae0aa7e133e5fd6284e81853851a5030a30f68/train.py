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
GRAD_CLIP_NORM = 2.0


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.layers(inputs), inplace=True)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 40),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(40, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model() -> nn.Module:
    return ImageClassifier()


class EMAAdamW(torch.optim.AdamW):
    def __init__(
        self,
        params,
        total_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self.total_steps = total_steps
        self.ema_start = max(1, int(0.70 * total_steps))
        self.completed_steps = 0
        self.ema_decay = 0.98
        self.ema_parameters: list[torch.Tensor] = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.completed_steps += 1

        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        if self.completed_steps == self.ema_start:
            self.ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
        elif self.completed_steps > self.ema_start:
            for average, parameter in zip(self.ema_parameters, parameters):
                average.lerp_(parameter.detach(), 1.0 - self.ema_decay)

        if (
            self.completed_steps >= self.total_steps
            and self.ema_parameters
        ):
            for parameter, average in zip(parameters, self.ema_parameters):
                parameter.copy_(average)

        return loss


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    return EMAAdamW(
        model.parameters(),
        total_steps=total_steps,
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=1.0e-3,
    )


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
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
    if progress < 0.05:
        multiplier = 0.2 + 0.8 * progress / 0.05
    else:
        cosine_progress = (progress - 0.05) / 0.95
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
