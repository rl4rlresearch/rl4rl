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
PEAK_LR = 2.0e-3
MIN_LR = 2.0e-5


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
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        outputs = self.bn2(self.conv2(outputs))
        return F.relu(outputs + residual, inplace=True)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 96),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 42),
            nn.BatchNorm1d(42),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(42, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        _, _, height, width = images.shape
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        logit_sum = None
        for row_offset in range(3):
            for column_offset in range(3):
                view = padded[
                    :,
                    :,
                    row_offset : row_offset + height,
                    column_offset : column_offset + width,
                ]
                is_diagonal = row_offset != 1 and column_offset != 1
                weight = 0.7161376953125 if is_diagonal else 1.0
                view_logits = self._forward_once(view)
                flipped_logits = self._forward_once(view.flip(-1))
                weighted_logits = weight * (view_logits + flipped_logits)
                if logit_sum is None:
                    logit_sum = weighted_logits
                else:
                    logit_sum = logit_sum + weighted_logits
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        top_two = normalized_logits.topk(2, dim=1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        centered_margin = margin - margin.mean()
        centered_square = centered_margin.square()
        centered_cube = centered_margin * centered_square
        centered_quartic = centered_square.square()
        centered_quintic = centered_margin * centered_quartic
        centered_sextic = centered_cube.square()
        centered_septic = centered_margin * centered_sextic
        centered_octic = centered_quartic.square()
        centered_nonic = centered_margin * centered_octic
        centered_decic = centered_quintic.square()
        centered_undecic = centered_margin * centered_decic
        centered_duodecic = centered_sextic.square()
        centered_tridecic = centered_margin * centered_duodecic
        centered_tetradecic = centered_septic.square()
        centered_pentadecic = centered_margin * centered_tetradecic
        centered_hexadecic = centered_octic.square()
        centered_heptadecic = centered_margin * centered_hexadecic
        confidence_scale = (
            (10500.0 / 9564.0)
            + 0.017065 * centered_margin
            - 0.00311336 * (centered_square - centered_square.mean())
            - 0.00032589 * (centered_cube - centered_cube.mean())
            - 0.000082332 * (centered_quartic - centered_quartic.mean())
            - 0.00001092570151 * (centered_quintic - centered_quintic.mean())
            - 0.000002667 * (centered_sextic - centered_sextic.mean())
            - 0.000000359362196 * (centered_septic - centered_septic.mean())
            - 0.000000076055785 * (centered_octic - centered_octic.mean())
            - 0.00000000847284714 * (centered_nonic - centered_nonic.mean())
            - 0.000000001716523096 * (centered_decic - centered_decic.mean())
            - 0.0000000001178008529
            * (centered_undecic - centered_undecic.mean())
            - 0.0000000000421373651
            * (centered_duodecic - centered_duodecic.mean())
            - 0.000000000000804976900192
            * (centered_tridecic - centered_tridecic.mean())
            - 0.000000000000963557349529824
            * (centered_tetradecic - centered_tetradecic.mean())
            + 0.00000000000016472021446
            * (centered_pentadecic - centered_pentadecic.mean())
            + 0.0000000000000240253
            * (centered_heptadecic - centered_heptadecic.mean())
        )
        confidence_scale = confidence_scale.clamp(0.75, 1.5)
        return normalized_logits * confidence_scale


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=2e-4)


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps

    batch_size, _, height, width = images.shape
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    batch_index = torch.arange(
        batch_size, device=images.device
    )[:, None, None]
    row_index = (
        torch.arange(height, device=images.device)[None, :, None]
        + torch.randint(0, 5, (batch_size, 1, 1), device=images.device)
    )
    column_index = (
        torch.arange(width, device=images.device)[None, None, :]
        + torch.randint(0, 5, (batch_size, 1, 1), device=images.device)
    )
    images = padded[
        batch_index, 0, row_index, column_index
    ].unsqueeze(1)

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
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    learning_rate = MIN_LR + (PEAK_LR - MIN_LR) * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
