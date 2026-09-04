"""Editable image-classification research program.

The verification harness owns the data split, example budget, evaluation, and
timing.  This file owns the learned model and the trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 48
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96 * 3 * 3 + 2 * 64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        shallow = self.stage1(images)
        middle = self.stage2(shallow)
        deep = self.stage3(middle)
        representation = torch.cat(
            (
                deep.flatten(1),
                middle.mean(dim=(2, 3)),
                middle.amax(dim=(2, 3)),
            ),
            dim=1,
        )
        return self.classifier(representation)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble = logits * 2.25
        for view in views[1:]:
            view_logits = self._classify(view)
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self._classify(view.flip(-1))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.25
            ensemble = ensemble + flipped_logits
        return ensemble / (12.5 * 0.9350)


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    batch = images.shape[0]
    choices = torch.randint(0, 6, (batch,), device=images.device)
    offset_y = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_x = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_y = torch.where(choices == 2, 0, offset_y)
    offset_y = torch.where(choices == 3, 2, offset_y)
    offset_x = torch.where(choices == 4, 0, offset_x)
    offset_x = torch.where(choices == 5, 2, offset_x)
    windows = F.pad(images, (1, 1, 1, 1)).unfold(2, 28, 1).unfold(3, 28, 1)
    indices = torch.arange(batch, device=images.device)
    images = windows[indices, :, offset_y, offset_x, :, :]
    flip = torch.rand(batch, 1, 1, 1, device=images.device) < 0.5
    images = torch.where(flip, images.flip(-1), images)
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
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup = 0.08
    if progress < warmup:
        learning_rate = 8e-4 + (3.0e-3 - 8e-4) * progress / warmup
    else:
        decay = (progress - warmup) / (1.0 - warmup)
        multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * decay))
        learning_rate = 3.0e-3 * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    update = step + 1
    in_tail = update >= int(0.8 * total_steps)
    sample_tail = in_tail and (update % 8 == 0 or update == total_steps)
    if sample_tail:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_average"):
                optimizer._tail_average = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._tail_average_count = 1
            else:
                optimizer._tail_average_count += 1
                weight = 1.0 / optimizer._tail_average_count
                for average, parameter in zip(
                    optimizer._tail_average, parameters
                ):
                    average.lerp_(parameter.detach(), weight)

            if update == total_steps:
                for parameter, average in zip(
                    parameters, optimizer._tail_average
                ):
                    parameter.lerp_(average, 0.5)
