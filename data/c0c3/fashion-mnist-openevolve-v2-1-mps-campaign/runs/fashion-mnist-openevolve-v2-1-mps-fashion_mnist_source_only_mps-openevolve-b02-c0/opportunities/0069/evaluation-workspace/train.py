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
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        height, width = images.shape[-2:]
        views = (
            images,
            padded[..., :height, 1 : width + 1],
            padded[..., 1 : height + 1, :width],
            padded[..., 1 : height + 1, 2 : width + 2],
            padded[..., 2 : height + 2, 1 : width + 1],
        )
        log_probabilities = []
        for view in views:
            paired_views = torch.cat((view, view.flip(-1)), dim=0)
            paired_log_probabilities = F.log_softmax(
                self._forward_once(paired_views), dim=1
            )
            log_probabilities.extend(paired_log_probabilities.chunk(2, dim=0))
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
        return 1.10 * ensemble_log_probabilities


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

    padding = 1
    padded = F.pad(images, (padding, padding, padding, padding), mode="replicate")
    height, width = images.shape[-2:]
    offset_draw = torch.randint(0, 13, (batch,), device=images.device)
    offsets_y = (
        1
        + ((offset_draw >= 7) & (offset_draw < 9)).long()
        - ((offset_draw >= 5) & (offset_draw < 7)).long()
    )
    offsets_x = (
        1
        + ((offset_draw >= 11) & (offset_draw < 13)).long()
        - ((offset_draw >= 9) & (offset_draw < 11)).long()
    )
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
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    smoothing = 0.03
    if progress > 0.75:
        smoothing *= (1.0 - progress) / 0.25
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)


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
    is_final_step = step + 1 >= total_steps
    if progress >= 0.95 and (step % 4 == 0 or is_final_step):
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        average_count = getattr(optimizer, "_late_average_count", 0)
        with torch.no_grad():
            if average_count == 0:
                optimizer._late_averaged_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
            else:
                update_weight = 1.0 / (average_count + 1)
                for average, parameter in zip(
                    optimizer._late_averaged_parameters, parameters
                ):
                    average.lerp_(parameter.detach(), update_weight)
            optimizer._late_average_count = average_count + 1

            if is_final_step:
                for parameter, average in zip(
                    parameters, optimizer._late_averaged_parameters
                ):
                    parameter.copy_(average)

    for group in optimizer.param_groups:
        group["lr"] = lr
