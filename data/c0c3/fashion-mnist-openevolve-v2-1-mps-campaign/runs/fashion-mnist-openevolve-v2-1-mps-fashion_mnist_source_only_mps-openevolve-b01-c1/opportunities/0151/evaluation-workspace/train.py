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
            nn.Conv2d(16, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.GELU(),
            nn.Conv2d(56, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)
        padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
        probability_sum = None
        central_probability_sum = None
        logit_sum = None
        central_logit_sum = None
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
                probabilities = F.softmax(logits, dim=-1)
                original_probs, flipped_probs = probabilities.chunk(2, dim=0)
                original_logits, flipped_logits = logits.chunk(2, dim=0)
                view_probabilities = original_probs + flipped_probs
                view_logits = original_logits + flipped_logits
                if probability_sum is None:
                    probability_sum = view_probabilities
                    logit_sum = view_logits
                else:
                    probability_sum = probability_sum + view_probabilities
                    logit_sum = logit_sum + view_logits
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    central_weight = (
                        (2 if offset_y == 2 else 1)
                        * (2 if offset_x == 2 else 1)
                    )
                    if central_probability_sum is None:
                        central_probability_sum = (
                            central_weight * view_probabilities
                        )
                        central_logit_sum = central_weight * view_logits
                    else:
                        central_probability_sum = (
                            central_probability_sum
                            + central_weight * view_probabilities
                        )
                        central_logit_sum = (
                            central_logit_sum
                            + central_weight * view_logits
                        )
        full_ensemble = probability_sum / 50.0
        central_ensemble = central_probability_sum / 32.0
        ensemble = 0.9 * full_ensemble + 0.1 * central_ensemble
        arithmetic_logits = 1.29834 * ensemble.clamp_min(1.0e-7).log()
        geometric_logits = (
            0.9 * logit_sum / 50.0
            + 0.1 * central_logit_sum / 32.0
        )
        geometric_logits = F.log_softmax(geometric_logits, dim=-1)
        return 0.93 * arithmetic_logits + 0.07 * geometric_logits


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
    smoothing = 0.009 + 0.0055 * (1.0 + math.cos(math.pi * progress))
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
        label_smoothing=smoothing,
    )
    central_individual_loss = F.cross_entropy(
        torch.cat((central_logits, flipped_central), dim=0),
        labels.repeat(2),
        label_smoothing=smoothing,
    )
    individual_loss = (
        0.9 * full_individual_loss + 0.1 * central_individual_loss
    )
    full_pair_loss = F.cross_entropy(
        torch.cat(
            (
                0.5 * (logits_a + flipped_a),
                0.5 * (logits_b + flipped_b),
            ),
            dim=0,
        ),
        labels.repeat(2),
        label_smoothing=smoothing,
    )
    central_pair_loss = F.cross_entropy(
        0.5 * (central_logits + flipped_central),
        labels,
        label_smoothing=smoothing,
    )
    pair_loss = 0.9 * full_pair_loss + 0.1 * central_pair_loss
    pair_weight = 0.375 - 0.125 * math.cos(math.pi * progress)
    view_loss = (
        (1.0 - pair_weight) * individual_loss + pair_weight * pair_loss
    )
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=smoothing,
    )
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * view_loss
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
        multiplier = 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
