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


class SpatialAttentionRefinement(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")

        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.projection = nn.Linear(channels, channels)

        relative_axis = torch.arange(-6, 7)
        delta_y, delta_x = torch.meshgrid(
            relative_axis,
            relative_axis,
            indexing="ij",
        )
        distance = delta_y.abs() + delta_x.abs()
        locality = torch.linspace(0.0, 0.45, heads).view(heads, 1, 1)
        self.relative_bias = nn.Parameter(
            -locality * distance.unsqueeze(0)
        )

        grid = torch.arange(7)
        coordinates = torch.stack(
            torch.meshgrid(grid, grid, indexing="ij")
        ).flatten(1)
        relative_offsets = (
            coordinates[:, :, None] - coordinates[:, None, :] + 6
        )
        self.register_buffer(
            "relative_offsets",
            relative_offsets,
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        token_count = height * width
        tokens = images.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)

        qkv = self.qkv(normalized).reshape(
            batch,
            token_count,
            3,
            self.heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention_logits = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) * self.scale
        position_bias = self.relative_bias[
            :,
            self.relative_offsets[0],
            self.relative_offsets[1],
        ]
        attention = F.softmax(
            attention_logits + position_bias.unsqueeze(0),
            dim=-1,
        )
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).reshape(
            batch,
            token_count,
            channels,
        )
        tokens = F.gelu(tokens + self.projection(context))
        return tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialAttentionRefinement(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 53),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(53, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

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
        probabilities = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * sum(crop_weights))
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.75317


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
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )


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
