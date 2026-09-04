"""Editable image-classification research program.

The verification harness owns the data split, example budget, evaluation, and
timing.  This file owns the learned model and the trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 32
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(28, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(56, 112, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(112, 112, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(112),
        )
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(112 * 3 * 3, 64),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(64, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stage1(images)
        features = self.stage2(features)
        features = self.stage3(features)
        features = F.relu(features + self.residual(features), inplace=True)
        return self.classifier(self.pool(features))

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
        log_prob_views = []
        for view in views:
            paired = torch.cat((view, view.flip(-1)), dim=0)
            logits, flipped_logits = self._forward_once(paired).chunk(2, dim=0)
            log_prob_views.append(F.log_softmax(logits, dim=1))
            log_prob_views.append(F.log_softmax(flipped_logits, dim=1))
        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(len(log_prob_views))


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if step < total_steps // 2:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        view = step % 5
        top = (1, 0, 2, 1, 1)[view]
        left = (1, 1, 1, 0, 2)[view]
        images = padded[:, :, top : top + 28, left : left + 28]
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
    update = step + 1
    progress = min(max(update / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    ema_start = max(1, total_steps // 2)
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    with torch.no_grad():
        if update == ema_start:
            optimizer._terminal_ema = [
                parameter.detach().clone() for parameter in parameters
            ]
            optimizer._terminal_ema_step = update
        elif update > ema_start and (
            update == total_steps
            or update - optimizer._terminal_ema_step >= 32
        ):
            elapsed = update - optimizer._terminal_ema_step
            decay = 0.995 ** elapsed
            for average, parameter in zip(
                optimizer._terminal_ema, parameters
            ):
                average.mul_(decay).add_(parameter, alpha=1.0 - decay)
            optimizer._terminal_ema_step = update
            if update == total_steps:
                for parameter, average in zip(
                    parameters, optimizer._terminal_ema
                ):
                    parameter.copy_(average)
