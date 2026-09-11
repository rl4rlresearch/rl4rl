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
GRAD_CLIP_NORM = 2.0

PEAK_LR = 3.0e-3
MIN_LR_RATIO = 0.05
WARMUP_FRACTION = 0.08


class ResidualSEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        hidden_channels = max(out_channels // 8, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(inputs)
        features = F.gelu(self.bn1(self.conv1(inputs)))
        features = self.bn2(self.conv2(features))
        features = features * self.channel_gate(features)
        return F.gelu(features + identity)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.features = nn.Sequential(
            ResidualSEBlock(32, 32),
            ResidualSEBlock(32, 48, stride=2),
            ResidualSEBlock(48, 48),
            ResidualSEBlock(48, 100, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(100, 10)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self.stem(images))
        return self.classifier(self.pool(features).flatten(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        crops = (
            padded[:, :, 1:29, 1:29],
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        logits = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
        for crop, weight in zip(crops, crop_weights):
            logits = logits + weight * self._forward_once(crop)
            logits = logits + weight * self._forward_once(crop.flip(-1))
        return logits / (2.0 * sum(crop_weights))


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=5e-4,
    )


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    batch = images.shape[0]

    flip_mask = torch.rand(
        (batch, 1, 1, 1), device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)

    padding = 1
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    translation = torch.randint(0, 7, (batch,), device=images.device)
    offsets_y = torch.tensor(
        (1, 1, 1, 0, 2, 1, 1), device=images.device
    )[translation].unsqueeze(1)
    offsets_x = torch.tensor(
        (1, 1, 1, 1, 1, 0, 2), device=images.device
    )[translation].unsqueeze(1)
    coordinates = torch.arange(28, device=images.device).unsqueeze(0)
    rows = coordinates + offsets_y
    columns = coordinates + offsets_x
    batch_indices = torch.arange(batch, device=images.device)[:, None, None]
    images = padded[:, 0][
        batch_indices,
        rows[:, :, None],
        columns[:, None, :],
    ].unsqueeze(1)

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
    if progress < WARMUP_FRACTION:
        multiplier = 0.2 + 0.8 * progress / WARMUP_FRACTION
    else:
        cosine_progress = (
            progress - WARMUP_FRACTION
        ) / (1.0 - WARMUP_FRACTION)
        multiplier = MIN_LR_RATIO + (
            1.0 - MIN_LR_RATIO
        ) * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier
