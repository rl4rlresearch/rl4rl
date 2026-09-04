"""Editable image-classification research program.

The verification harness owns the data split, example budget, evaluation, and
timing.  This file owns the learned model and the trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weight * maximum + (1.0 - max_weight) * average


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(
                80, 80, kernel_size=3, padding=1, groups=80, bias=False
            ),
            nn.BatchNorm2d(80),
            nn.GELU(),
            nn.Conv2d(80, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
        )
        self.aggregate_local = nn.Sequential(
            nn.Conv2d(
                80, 40, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(40),
            nn.GELU(),
        )
        self.aggregate_context = nn.Sequential(
            nn.Conv2d(
                80,
                24,
                kernel_size=3,
                stride=2,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )

    @staticmethod
    def _image_basis(images: torch.Tensor) -> torch.Tensor:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        local_mean = F.avg_pool2d(padded, kernel_size=3, stride=1)
        local_contrast = images - local_mean

        gradient_x = 0.125 * (
            padded[:, :, :-2, 2:]
            + 2.0 * padded[:, :, 1:-1, 2:]
            + padded[:, :, 2:, 2:]
            - padded[:, :, :-2, :-2]
            - 2.0 * padded[:, :, 1:-1, :-2]
            - padded[:, :, 2:, :-2]
        )
        gradient_y = 0.125 * (
            padded[:, :, 2:, :-2]
            + 2.0 * padded[:, :, 2:, 1:-1]
            + padded[:, :, 2:, 2:]
            - padded[:, :, :-2, :-2]
            - 2.0 * padded[:, :, :-2, 1:-1]
            - padded[:, :, :-2, 2:]
        )
        edge_energy = torch.sqrt(
            gradient_x.square() + gradient_y.square() + 1.0e-6
        )
        return torch.cat(
            (
                images,
                local_contrast,
                gradient_x,
                gradient_y,
                edge_energy,
            ),
            dim=1,
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        features = torch.cat(
            (
                self.aggregate_local(features),
                self.aggregate_context(features),
            ),
            dim=1,
        )
        return self.classifier(features)

    def _flip_average(self, images: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self._forward_once(images)
            + self._forward_once(images.flip(-1))
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        height, width = images.shape[-2:]
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        logits = 0.363447265625 * self._flip_average(images)

        for delta_y in range(-2, 3):
            for delta_x in range(-2, 3):
                if delta_y == 0 and delta_x == 0:
                    continue

                shifted = padded[
                    :,
                    :,
                    2 + delta_y : 2 + delta_y + height,
                    2 + delta_x : 2 + delta_x + width,
                ]
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10875
                    elif delta_y == 0:
                        weight = 0.0695263671875
                    else:
                        weight = 0.045
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
                logits = logits + weight * self._flip_average(shifted)
        return 1.253 * logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.96),
    )


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    height, width = images.shape[-2:]
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    translation_weights = images.new_tensor(
        (
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.09000, 0.36000, 0.09000, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
        )
    )
    flat_offsets = torch.multinomial(
        translation_weights, images.size(0), replacement=True
    )
    offsets = torch.stack(
        (flat_offsets.div(5, rounding_mode="floor"), flat_offsets.remainder(5)),
        dim=1,
    )

    rows = offsets[:, 0, None] + torch.arange(height, device=images.device)
    row_index = rows[:, None, :, None].expand(
        -1, images.size(1), -1, padded.size(3)
    )
    images = padded.gather(2, row_index)

    columns = offsets[:, 1, None] + torch.arange(width, device=images.device)
    column_index = columns[:, None, None, :].expand(
        -1, images.size(1), height, -1
    )
    images = images.gather(3, column_index)

    return images, labels


def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = min(step / max(total_steps, 1), 1.0)
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
    individual_loss = F.cross_entropy(
        paired_logits, paired_labels, label_smoothing=smoothing
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=smoothing,
    )
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
