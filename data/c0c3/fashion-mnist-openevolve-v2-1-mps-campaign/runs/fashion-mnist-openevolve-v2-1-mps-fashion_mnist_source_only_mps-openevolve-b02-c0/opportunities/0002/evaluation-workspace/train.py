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
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
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
            self.skip = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 48, stride=2),
            ResidualBlock(48, 48),
            ResidualBlock(48, 64, stride=2),
            ResidualBlock(64, 64),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.10),
            nn.Linear(64, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits
        flipped_logits = self._forward_once(images.flip(-1))
        return torch.logaddexp(
            F.log_softmax(logits, dim=1),
            F.log_softmax(flipped_logits, dim=1),
        ) - math.log(2.0)


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=6.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    batch = images.shape[0]

    flip_mask = torch.rand(batch, 1, 1, 1, device=images.device) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)

    padding = 2
    padded = F.pad(images, (padding, padding, padding, padding), mode="replicate")
    height, width = images.shape[-2:]
    offsets_y = torch.randint(0, 2 * padding + 1, (batch,), device=images.device)
    offsets_x = torch.randint(0, 2 * padding + 1, (batch,), device=images.device)
    batch_index = torch.arange(batch, device=images.device)[:, None, None]
    rows = offsets_y[:, None, None] + torch.arange(
        height, device=images.device
    )[None, :, None]
    columns = offsets_x[:, None, None] + torch.arange(
        width, device=images.device
    )[None, None, :]
    images = padded[:, 0][batch_index, rows, columns].unsqueeze(1).contiguous()
    return images, labels


def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    warmup_fraction = 0.05
    start_lr = 6.0e-4
    peak_lr = 3.0e-3
    end_lr = 1.0e-4
    if progress < warmup_fraction:
        lr = start_lr + (peak_lr - start_lr) * progress / warmup_fraction
    else:
        decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        lr = end_lr + 0.5 * (peak_lr - end_lr) * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
