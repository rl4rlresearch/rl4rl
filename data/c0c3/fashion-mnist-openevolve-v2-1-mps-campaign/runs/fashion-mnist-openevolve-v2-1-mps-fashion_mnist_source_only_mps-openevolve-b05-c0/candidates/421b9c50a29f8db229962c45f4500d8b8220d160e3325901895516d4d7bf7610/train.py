"""Editable image-classification research program.

The verification harness owns the data split, example budget, evaluation, and
timing.  This file owns the learned model and the trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 32
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
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.depthwise(inputs)
        features = F.gelu(self.depthwise_norm(features))
        features = self.pointwise_norm(self.pointwise(features))
        return F.gelu(inputs + features)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                32, 56, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(56),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                56, 96, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualDepthwiseBlock(96),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 199),
            nn.GELU(),
            nn.Linear(199, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.416375
        return logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.5e-3, weight_decay=1e-4
    )
    optimizer._model = model
    return optimizer


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
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    first_step = getattr(optimizer, "_first_training_step", None)
    if first_step is None:
        first_step = step
        optimizer._first_training_step = step
    completed_steps = step - first_step + 1

    if completed_steps >= total_steps // 2:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        ema_parameters = getattr(optimizer, "_ema_parameters", None)
        with torch.no_grad():
            ema_start = total_steps // 2
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_parameters = ema_parameters
                optimizer._ema_buffer_starts = [
                    buffer.detach().clone()
                    for buffer in optimizer._model.buffers()
                    if torch.is_floating_point(buffer)
                ]
            elif completed_steps >= total_steps:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.06984375)
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.03)

            if completed_steps >= total_steps:
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
                buffers = [
                    buffer
                    for buffer in optimizer._model.buffers()
                    if torch.is_floating_point(buffer)
                ]
                for buffer, start in zip(
                    buffers, optimizer._ema_buffer_starts
                ):
                    buffer.lerp_(start, 0.25)
