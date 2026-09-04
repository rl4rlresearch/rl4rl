"""Editable image-classification research program.

The verification harness owns the data split, example budget, evaluation, and
timing.  This file owns the learned model and the trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 96
GRAD_CLIP_NORM = 1.0
PEAK_LR = 3.3e-3


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = F.gelu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return F.gelu(outputs + residual)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
        )
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        spatial_mean = feature_map.mean(dim=(2, 3))
        spatial_std = feature_map.var(
            dim=(2, 3), unbiased=False
        ).add(1e-4).sqrt()
        spatial_max = feature_map.amax(dim=(2, 3))
        channel_gate = 2.0 * torch.sigmoid(
            spatial_mean * self.recalibration_scale
            + self.recalibration_bias
        )
        gated_feature_map = (
            feature_map * channel_gate[:, :, None, None]
        )
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
        statistics = statistics * channel_gate.repeat(1, 3)
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(gated_feature_map) + residual_logits

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
        center_weight = 1.81732177734375
        view_logits = [
            self._forward_once(views[0]),
            self._forward_once(views[0].flip(-1)),
        ]
        for view in views[1:]:
            view_logits.append(self._forward_once(view))
            view_logits.append(self._forward_once(view.flip(-1)))

        logits = center_weight * (
            view_logits[0] + view_logits[1]
        )
        for translated_logits in view_logits[2:]:
            logits = logits + translated_logits
        logits = logits / (
            2.0 * center_weight + 2.0 * (len(views) - 1)
        )

        predictions = torch.stack(view_logits).argmax(dim=2)
        ensemble_prediction = logits.argmax(dim=1)
        agreement = predictions.eq(
            ensemble_prediction.unsqueeze(0)
        ).float().mean(dim=0)
        agreement = agreement - agreement.mean()
        confidence_scale = torch.exp(0.12 * agreement)
        return 1.0671112 * confidence_scale[:, None] * logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
    optimizer.ema_pairs = None
    optimizer.ema_buffer_sources = [
        buffer
        for buffer in model.buffers()
        if torch.is_floating_point(buffer)
    ]
    optimizer.ema_buffer_pairs = None
    return optimizer


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    batch_size = images.shape[0]
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    windows = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    view_ids = torch.randint(0, 6, (batch_size,), device=images.device)
    offset_table = torch.tensor(
        ((1, 1), (1, 1), (0, 1), (2, 1), (1, 0), (1, 2)),
        device=images.device,
    )
    offsets = offset_table[view_ids]
    batch_indices = torch.arange(batch_size, device=images.device)
    images = windows[
        batch_indices, :, offsets[:, 0], offsets[:, 1]
    ]
    flip_mask = torch.rand(batch_size, device=images.device) < 0.5
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
    return F.cross_entropy(model(images), labels)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    with torch.no_grad():
        if optimizer.ema_pairs is None:
            optimizer.ema_pairs = [
                (parameter, parameter.detach().clone())
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
        else:
            updates = step + 1
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
            for parameter, average in optimizer.ema_pairs:
                average.lerp_(parameter, 1.0 - ema_decay)

        if step + 1 >= total_steps:
            for parameter, average in optimizer.ema_pairs:
                parameter.copy_(average)

    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.2 + 0.8 * progress / warmup_fraction
    else:
        decay_progress = (
            (progress - warmup_fraction) / (1.0 - warmup_fraction)
        )
        multiplier = 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier
