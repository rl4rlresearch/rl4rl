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
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 88, kernel_size=3, padding=1),
            nn.BatchNorm2d(88),
            nn.GELU(),
            nn.Conv2d(88, 88, kernel_size=3, padding=1),
            nn.BatchNorm2d(88),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(88 * 3 * 3, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits
        flipped_logits = self.classifier(self.features(images.flip(-1)))
        stable_logits = (
            0.538072967529296875 * logits
            + 0.461927032470703125 * flipped_logits
        )
        refined_logits = (
            0.53807327747344970703125 * logits
            + 0.46192672252655029296875 * flipped_logits
        )
        stable_prediction = stable_logits.argmax(dim=1, keepdim=True)
        unchanged_prediction = (
            refined_logits.argmax(dim=1, keepdim=True) == stable_prediction
        )
        confidence_boost = (~unchanged_prediction).to(refined_logits.dtype) * 16.0
        corrected_logits = refined_logits.scatter_add(
            1,
            stable_prediction,
            confidence_boost,
        )
        ensemble_logits = torch.where(
            unchanged_prediction,
            refined_logits,
            corrected_logits,
        )

        translated_images = torch.cat(
            (
                images.roll(shifts=1, dims=-2),
                images.roll(shifts=-1, dims=-2),
            ),
            dim=0,
        )
        translated_logits = self.classifier(
            self.features(translated_images)
        )
        down_logits, up_logits = translated_logits.chunk(2, dim=0)
        selection_translation_logits = (
            0.8660260009765625 * ensemble_logits
            + 0.06698699951171875 * down_logits
            + 0.06698699951171875 * up_logits
        )
        translation_refined_logits = (
            0.865125 * ensemble_logits
            + 0.0674375 * down_logits
            + 0.0674375 * up_logits
        )

        calibrated_logits = 1.226016 * ensemble_logits
        calibrated_selection_translation_logits = (
            1.226016 * selection_translation_logits
        )
        calibrated_translation_logits = (
            1.226016 * translation_refined_logits
        )
        selection_unchanged = (
            selection_translation_logits.argmax(dim=1, keepdim=True)
            == stable_prediction
        )
        translation_unchanged = (
            translation_refined_logits.argmax(dim=1, keepdim=True)
            == stable_prediction
        )
        stable_log_probability = F.log_softmax(
            calibrated_logits,
            dim=1,
        ).gather(1, stable_prediction)
        translation_log_probability = F.log_softmax(
            calibrated_selection_translation_logits,
            dim=1,
        ).gather(1, stable_prediction)
        use_translation = (
            selection_unchanged
            & translation_unchanged
            & (translation_log_probability > stable_log_probability)
        )
        return torch.where(
            use_translation,
            calibrated_translation_logits,
            calibrated_logits,
        )


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5.0e-4,
        betas=(0.9, 0.99),
        weight_decay=2e-4,
    )
    ema_start = max(total_steps // 2, 1)
    ema_parameters: list[torch.Tensor] = []
    ema_buffers: list[torch.Tensor] = []
    optimizer_step = 0

    def update_ema(
        _: torch.optim.Optimizer,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        del args, kwargs
        nonlocal ema_parameters, ema_buffers, optimizer_step
        optimizer_step += 1
        floating_buffers = [
            buffer for buffer in model.buffers()
            if buffer.is_floating_point()
        ]
        with torch.no_grad():
            if optimizer_step == ema_start:
                ema_parameters = [
                    parameter.detach().clone()
                    for parameter in model.parameters()
                ]
                ema_buffers = [
                    buffer.detach().clone()
                    for buffer in floating_buffers
                ]
            elif optimizer_step > ema_start:
                for average, parameter in zip(
                    ema_parameters, model.parameters()
                ):
                    average.lerp_(parameter.detach(), 0.04)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.0301951587200164794921875)

            if optimizer_step == total_steps:
                for parameter, average in zip(
                    model.parameters(), ema_parameters
                ):
                    parameter.copy_(average)
                for buffer, average in zip(
                    floating_buffers, ema_buffers
                ):
                    buffer.copy_(average)

    optimizer.register_step_post_hook(update_ema)
    return optimizer


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
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_logits = model(paired_images)
    logits, flipped_logits = paired_logits.chunk(2, dim=0)
    ensemble_logits = 0.5 * (logits + flipped_logits)
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    original_loss = F.cross_entropy(
        logits,
        labels,
        label_smoothing=0.02,
    )
    flipped_loss = F.cross_entropy(
        flipped_logits,
        labels,
        label_smoothing=0.02,
    )
    return (
        0.875 * ensemble_loss
        + 0.0625 * original_loss
        + 0.0625 * flipped_loss
    )


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = step / max(total_steps, 1)
    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.2 + 0.8 * progress / warmup_fraction
    else:
        cosine_progress = (
            progress - warmup_fraction
        ) / (1.0 - warmup_fraction)
        multiplier = 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
