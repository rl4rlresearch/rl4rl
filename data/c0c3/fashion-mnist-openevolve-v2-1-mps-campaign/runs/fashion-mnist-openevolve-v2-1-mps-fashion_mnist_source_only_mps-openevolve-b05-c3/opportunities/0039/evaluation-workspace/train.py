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
BASE_LR = 3.0e-3


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = ((1, 32), (32, 32), (32, 64), (64, 64), (64, 96), (96, 96))
        for index, (in_channels, out_channels) in enumerate(channels):
            layers.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            )
            if index in (1, 3, 5):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        logit_sum = logits
        logit_sum.add_(self._forward_once(images.flip(-1)))

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in shifted_views:
            logit_sum.add_(self._forward_once(view))
            logit_sum.add_(self._forward_once(view.flip(-1)))

        return logit_sum / 10.0


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR * 0.2,
        weight_decay=1.5e-4,
    )
    optimizer._ema_tensors = [
        tensor
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.is_floating_point()
    ]
    optimizer._ema_values = [
        tensor.detach().clone() for tensor in optimizer._ema_tensors
    ]
    optimizer._ema_started = False
    return optimizer


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_indices = torch.arange(images.shape[0], device=images.device)
    if step * 64 < total_steps * 39:
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
        offsets_y = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
        offsets_x = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
    else:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
        directions = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
        offsets_y = (
            1 + (directions == 2).long() - (directions == 1).long()
        )
        offsets_x = (
            1 + (directions == 4).long() - (directions == 3).long()
        )
    images = crops[batch_indices, :, offsets_y, offsets_x]
    flip_mask = torch.rand(
        (images.shape[0], 1, 1, 1), device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
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
    warmup_steps = max(1, total_steps // 20)
    if step < warmup_steps:
        multiplier = 0.2 + 0.8 * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        multiplier = 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = BASE_LR * multiplier

    with torch.no_grad():
        if not optimizer._ema_started:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.copy_(tensor)
            optimizer._ema_started = True
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)

        if step >= total_steps:
            for tensor, average in zip(
                optimizer._ema_tensors, optimizer._ema_values
            ):
                tensor.copy_(average)
