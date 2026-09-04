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
        self.residual = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 80),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Linear(80, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)
        padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
        logits_sum = None
        central_logits_sum = None
        for offset_y in range(5):
            for offset_x in range(5):
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
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    if central_logits_sum is None:
                        central_logits_sum = view_logits
                    else:
                        central_logits_sum = central_logits_sum + view_logits
        full_ensemble = logits_sum / 50.0
        central_ensemble = central_logits_sum / 18.0
        return 1.29834 * (0.9 * full_ensemble + 0.1 * central_ensemble)


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
    return images, labels


def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    offsets = torch.randint(0, 5, (2,))
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    central_offsets = torch.randint(1, 4, (2,))
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    view_a = padded[
        :,
        :,
        offset_y : offset_y + 28,
        offset_x : offset_x + 28,
    ]
    view_b = padded[
        :,
        :,
        4 - offset_y : 4 - offset_y + 28,
        4 - offset_x : 4 - offset_x + 28,
    ]
    central_view = padded[
        :,
        :,
        central_y : central_y + 28,
        central_x : central_x + 28,
    ]
    logits = model(
        torch.cat(
            (
                view_a,
                view_b,
                view_a.flip(-1),
                view_b.flip(-1),
                central_view,
                central_view.flip(-1),
            ),
            dim=0,
        )
    )
    (
        logits_a,
        logits_b,
        flipped_a,
        flipped_b,
        central_logits,
        flipped_central,
    ) = logits.chunk(6, dim=0)
    full_individual_loss = F.cross_entropy(
        torch.cat((logits_a, logits_b, flipped_a, flipped_b), dim=0),
        labels.repeat(4),
        label_smoothing=0.02,
    )
    central_individual_loss = F.cross_entropy(
        torch.cat((central_logits, flipped_central), dim=0),
        labels.repeat(2),
        label_smoothing=0.02,
    )
    individual_loss = (
        0.9 * full_individual_loss + 0.1 * central_individual_loss
    )
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )


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

    completed_steps = step + 1
    if completed_steps % 32 == 0 or completed_steps == total_steps:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_ema_parameters"):
                optimizer._ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_step = completed_steps
            else:
                elapsed_steps = completed_steps - optimizer._ema_step
                decay = 0.99 ** elapsed_steps
                torch._foreach_mul_(optimizer._ema_parameters, decay)
                torch._foreach_add_(
                    optimizer._ema_parameters,
                    parameters,
                    alpha=1.0 - decay,
                )
                optimizer._ema_step = completed_steps
            if completed_steps == total_steps:
                for parameter, averaged_parameter in zip(
                    parameters, optimizer._ema_parameters
                ):
                    parameter.copy_(averaged_parameter)
