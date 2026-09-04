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


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 58),
            nn.BatchNorm1d(58),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(58, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

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

        def collect_log_probabilities() -> list[torch.Tensor]:
            outputs = []
            for view in views:
                outputs.append(
                    F.log_softmax(self._forward_once(view), dim=1)
                )
                outputs.append(
                    F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
                )
            return outputs

        log_probabilities = collect_log_probabilities()
        ema_state = getattr(self, "_ema_state", None)
        if ema_state is not None:
            live_state = dict(self.named_parameters())
            live_state.update(dict(self.named_buffers()))
            backups = {}
            with torch.no_grad():
                for name, average in ema_state.items():
                    tensor = live_state[name]
                    backups[name] = tensor.detach().clone()
                    tensor.copy_(average)
            try:
                log_probabilities.extend(collect_log_probabilities())
            finally:
                with torch.no_grad():
                    for name, backup in backups.items():
                        live_state[name].copy_(backup)

        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=3e-4,
    )
    optimizer._ema_model = model
    optimizer._ema_state = None
    return optimizer


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    windows = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    batch_index = torch.arange(images.shape[0], device=images.device)
    cardinal_offsets = torch.tensor(
        ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)),
        device=images.device,
    )
    choices = torch.randint(
        0, cardinal_offsets.shape[0], (images.shape[0],), device=images.device
    )
    shifts = cardinal_offsets[choices]
    images = windows[batch_index, :, shifts[:, 0], shifts[:, 1]]
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
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
    ema_model = getattr(optimizer, "_ema_model", None)
    if ema_model is not None and step >= total_steps // 2:
        live_state = dict(ema_model.named_parameters())
        live_state.update(dict(ema_model.named_buffers()))
        floating_state = {
            name: tensor
            for name, tensor in live_state.items()
            if tensor.is_floating_point()
        }
        if optimizer._ema_state is None:
            optimizer._ema_state = {
                name: tensor.detach().clone()
                for name, tensor in floating_state.items()
            }
        else:
            decay = 0.985
            with torch.no_grad():
                for name, average in optimizer._ema_state.items():
                    average.mul_(decay).add_(
                        floating_state[name].detach(), alpha=1.0 - decay
                    )
        ema_model._ema_state = optimizer._ema_state

    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    warmup_fraction = 0.08
    start_lr = 3e-4
    peak_lr = 1.8e-3
    end_lr = 7e-5
    if progress < warmup_fraction:
        lr = start_lr + (peak_lr - start_lr) * progress / warmup_fraction
    else:
        cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        lr = end_lr + (peak_lr - end_lr) * cosine
    for group in optimizer.param_groups:
        group["lr"] = lr
