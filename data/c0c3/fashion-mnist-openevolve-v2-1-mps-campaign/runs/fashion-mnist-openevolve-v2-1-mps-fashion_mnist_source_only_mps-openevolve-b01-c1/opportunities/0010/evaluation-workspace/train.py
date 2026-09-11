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


class _GlobalChannelGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(channels // 8, 8)
        self.reduce = nn.Linear(channels, hidden_channels)
        self.expand = nn.Linear(hidden_channels, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = F.adaptive_avg_pool2d(features, 1).flatten(1)
        gates = self.expand(F.gelu(self.reduce(context)))
        gates = 2.0 * torch.sigmoid(gates).unsqueeze(-1).unsqueeze(-1)
        return features * gates


class _ContextResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bottleneck_channels: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(
                input_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                output_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )
        self.gate = _GlobalChannelGate(output_channels)
        if input_channels == output_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.expand(self.spatial(self.reduce(features)))
        residual = self.gate(residual)
        return F.gelu(residual + self.shortcut(features))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.blocks = nn.ModuleList(
            (
                _ContextResidualBlock(32, 72, 56, dilation=1),
                _ContextResidualBlock(72, 72, 56, dilation=2),
                _ContextResidualBlock(72, 72, 56, dilation=1),
                _ContextResidualBlock(72, 72, 56, dilation=2),
            )
        )
        self.spatial_attention = nn.Conv2d(
            72,
            4,
            kernel_size=1,
        )
        self.classifier = nn.Sequential(
            nn.Linear(72 * 9, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        for block in self.blocks:
            features = block(features)

        coarse_shape = F.adaptive_avg_pool2d(features, 2).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)

        attention = self.spatial_attention(features).flatten(2)
        attention = F.softmax(attention, dim=-1)
        feature_tokens = features.flatten(2).unsqueeze(1)
        attended_features = (
            feature_tokens * attention.unsqueeze(2)
        ).sum(dim=-1).flatten(1)

        descriptor = torch.cat(
            (coarse_shape, peak_features, attended_features),
            dim=1,
        )
        return self.classifier(descriptor)

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
        return logits_sum / 18.0


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
