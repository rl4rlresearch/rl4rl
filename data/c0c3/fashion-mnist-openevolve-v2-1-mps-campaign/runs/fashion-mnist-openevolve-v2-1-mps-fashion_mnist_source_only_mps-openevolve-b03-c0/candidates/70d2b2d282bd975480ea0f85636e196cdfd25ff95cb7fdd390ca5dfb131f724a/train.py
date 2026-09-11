"""Editable image-classification research program.

The verification harness owns the data split, example budget, evaluation, and
timing.  This file owns the learned model and the trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 64
GRAD_CLIP_NORM = 2.0


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.layers(inputs), inplace=True)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 44),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(44, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
            translated_images = torch.cat(
                (
                    padded[:, :, 1:29, 0:28],
                    padded[:, :, 1:29, 2:30],
                    padded[:, :, 0:28, 1:29],
                    padded[:, :, 2:30, 1:29],
                ),
                dim=0,
            )
            translated_view_logits = self.classifier(
                self.features(translated_images)
            ).reshape(4, images.shape[0], 10)
            translated_logits = translated_view_logits.mean(dim=0)
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
            base_predictions = logits.argmax(dim=1)
            preserving_predictions = preserving_logits.argmax(dim=1)
            correction_predictions = correction_logits.argmax(dim=1)
            preserves_argmax = preserving_predictions.eq(base_predictions)
            unanimous_correction = (
                correction_predictions.ne(base_predictions)
                & translated_view_logits.argmax(dim=2).eq(
                    correction_predictions.unsqueeze(0)
                ).all(dim=0)
            )
            logits = torch.where(
                unanimous_correction.unsqueeze(1),
                correction_logits,
                torch.where(
                    preserves_argmax.unsqueeze(1),
                    preserving_logits,
                    logits,
                ),
            )
            logits = logits / 0.7381525
        return logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=1.0e-3,
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
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    if progress < 0.05:
        multiplier = 0.2 + 0.8 * progress / 0.05
    else:
        cosine_progress = (progress - 0.05) / 0.95
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
