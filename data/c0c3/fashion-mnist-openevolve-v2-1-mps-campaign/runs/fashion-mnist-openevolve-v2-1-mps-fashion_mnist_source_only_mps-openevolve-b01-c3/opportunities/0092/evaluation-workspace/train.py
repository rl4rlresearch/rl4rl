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


class MultiScaleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = channels // 2
        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat(
            (self.local(features), self.context(features)),
            dim=1,
        )
        return F.gelu(features + self.fuse(mixed))


class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features + self.refine(features))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.early = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.mid_context = MultiScaleBlock(64)
        self.pool2 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.early(features))
        features = self.down1(self.pool1(features))
        features = self.mid_context(features)
        features = self.down2(self.pool2(features))
        features = self.late_context(features)
        features = self.late_refinement(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._predict(images)

        height, width = images.shape[-2:]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = []
        for row_offset, col_offset in (
            (1, 1),
            (0, 1),
            (2, 1),
            (1, 0),
            (1, 2),
        ):
            view = padded[
                :,
                :,
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
            views.append(view)
            views.append(view.flip(-1))

        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.546875 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.546875
        return 1.225825 * pooled_logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5.0e-4,
        weight_decay=2e-4,
    )
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
    optimizer.tail_average_weight_sum = 0.0
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return optimizer


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
    row_offsets = torch.tensor(
        (1, 0, 2, 1, 1), device=images.device
    )[positions].unsqueeze(1)
    col_offsets = torch.tensor(
        (1, 1, 1, 0, 2), device=images.device
    )[positions].unsqueeze(1)
    rows = row_offsets + torch.arange(height, device=images.device).unsqueeze(0)
    row_index = rows[:, None, :, None].expand(
        batch, channels, height, padded.shape[-1]
    )
    images = padded.gather(2, row_index)

    cols = col_offsets + torch.arange(width, device=images.device).unsqueeze(0)
    col_index = cols[:, None, None, :].expand(batch, channels, height, width)
    images = images.gather(3, col_index)

    flip_mask = torch.rand(batch, device=images.device) < 0.5
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
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.2 + 0.8 * progress / warmup_fraction
    else:
        decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        multiplier = 0.04 + 0.96 * 0.5 * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    tail_index = step + 2 - optimizer.tail_average_start
    tail_length = total_steps - optimizer.tail_average_start + 1
    if (
        step + 1 >= optimizer.tail_average_start
        and (tail_index == 1 or tail_index % 2 == 0)
    ):
        recency_power = 1.0
        if tail_index == 1:
            sample_weight = 1.0
        elif tail_index == 2:
            sample_weight = (
                tail_index ** recency_power
                + 0.5 * (tail_index + 1) ** recency_power
            )
        elif tail_index == tail_length:
            sample_weight = (
                tail_index ** recency_power
                + 0.5 * (tail_index - 1) ** recency_power
            )
        else:
            sample_weight = (
                tail_index ** recency_power
                + 0.5 * (tail_index - 1) ** recency_power
                + 0.5 * (tail_index + 1) ** recency_power
            )

        optimizer.tail_average_weight_sum += sample_weight
        average_weight = sample_weight / optimizer.tail_average_weight_sum
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.tail_average_parameters,
                parameters,
            ):
                average.lerp_(parameter, average_weight)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer.tail_average_parameters,
                ):
                    parameter.copy_(average)
