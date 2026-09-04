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


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        features = self.bn2(self.conv2(features))
        return F.relu(features + inputs, inplace=True)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            ResidualBlock(24),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ResidualBlock(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            ResidualBlock(80),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.10),
            nn.Linear(80 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(64, 10),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay_parameters = []
    no_decay_parameters = []
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": 2.0e-4},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=4.0e-4,
    )


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    batch_size = images.shape[0]
    theta = images.new_zeros((batch_size, 2, 3))

    flip = torch.where(
        torch.rand(batch_size, device=images.device) < 0.5,
        -torch.ones(batch_size, device=images.device),
        torch.ones(batch_size, device=images.device),
    ).to(images.dtype)
    shifts = torch.randint(
        -2, 3, (batch_size, 2), device=images.device
    ).to(images.dtype)

    theta[:, 0, 0] = flip
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = shifts[:, 0] * (2.0 / images.shape[-1])
    theta[:, 1, 2] = shifts[:, 1] * (2.0 / images.shape[-2])

    grid = F.affine_grid(theta, images.shape, align_corners=False)
    images = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
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
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.08
    initial_lr = 4.0e-4
    peak_lr = 3.0e-3
    final_lr = 1.0e-4

    if progress < warmup_fraction:
        ratio = progress / warmup_fraction
        learning_rate = initial_lr + ratio * (peak_lr - initial_lr)
    else:
        ratio = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
        learning_rate = final_lr + (peak_lr - final_lr) * cosine

    for group in optimizer.param_groups:
        group["lr"] = learning_rate
