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


class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        return self.pool(F.silu(features + refinement, inplace=True))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualStage(1, 32),
            ResidualStage(32, 64),
            ResidualStage(64, 96),
        )
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

        probability_sum = F.softmax(logits, dim=1)
        probability_sum.add_(
            F.softmax(self._forward_once(images.flip(-1)), dim=1)
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in shifted_views:
            probability_sum.add_(F.softmax(self._forward_once(view), dim=1))
            probability_sum.add_(
                F.softmax(self._forward_once(view.flip(-1)), dim=1)
            )

        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.055)


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR * 0.2,
        weight_decay=1.5e-4,
    )
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    optimizer._ema_tensors = [
        tensor
        for tensor in parameters + buffers
        if tensor.is_floating_point()
    ]
    buffer_ids = {id(tensor) for tensor in buffers}
    optimizer._ema_is_buffer = [
        id(tensor) in buffer_ids for tensor in optimizer._ema_tensors
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
    if step * 32 < total_steps * 19:
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
            terminal_phase = step * 32 >= total_steps * 19
            for average, tensor, is_buffer in zip(
                optimizer._ema_values,
                optimizer._ema_tensors,
                optimizer._ema_is_buffer,
            ):
                if terminal_phase:
                    ema_rate = 0.0325 if is_buffer else 0.015
                else:
                    ema_rate = 0.005
                average.lerp_(tensor, ema_rate)

        if step >= total_steps:
            for tensor, average in zip(
                optimizer._ema_tensors, optimizer._ema_values
            ):
                tensor.copy_(average)
