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
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.transition = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.MaxPool2d(2)
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )

    def _forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = self.pool(features)
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        return self.pool(features)

    def _classify_views(
        self,
        features: torch.Tensor,
        flipped_features: torch.Tensor,
    ) -> torch.Tensor:
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size = images.shape[0]
            paired_images = torch.cat(
                (images, torch.flip(images, dims=(-1,))),
                dim=0,
            )
            paired_features = self._forward_features(paired_images)
            features = paired_features[:batch_size]
            flipped_features = paired_features[batch_size:]
        else:
            batch_size = images.shape[0]
            shifted_up = F.pad(
                images, (0, 0, 0, 1), mode="replicate"
            )[:, :, 1:, :]
            shifted_down = F.pad(
                images, (0, 0, 1, 0), mode="replicate"
            )[:, :, :-1, :]
            view_images = torch.cat(
                (images, shifted_up, shifted_down),
                dim=0,
            )
            view_count = view_images.shape[0]
            paired_images = torch.cat(
                (
                    view_images,
                    torch.flip(view_images, dims=(-1,)),
                ),
                dim=0,
            )
            paired_features = self._forward_features(paired_images)
            features = paired_features[:view_count]
            flipped_features = paired_features[view_count:]

        logits = self._classify_views(features, flipped_features)
        if self.training:
            return logits
        logits = logits.reshape(3, batch_size, 10).mean(dim=0)
        return 1.2112 * logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.1e-3,
        betas=(0.9, 0.99),
        weight_decay=2e-4,
    )
    optimizer.ema_model = model
    optimizer.ema_state = None
    optimizer.ema_parameter_names = {
        name for name, _ in model.named_parameters()
    }
    return optimizer


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
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
    progress = min(step / max(total_steps, 1), 1.0)
    dropout_decay = max(progress - 0.50, 0.0) / 0.50
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    decay_progress = max(progress - 0.10, 0.0) / 0.90
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
    for group in optimizer.param_groups:
        group["lr"] = 2.1e-3 * multiplier

    if progress >= 0.50:
        current_state = optimizer.ema_model.state_dict()
        with torch.no_grad():
            if optimizer.ema_state is None:
                optimizer.ema_state = {
                    name: value.detach().clone()
                    for name, value in current_state.items()
                }
            else:
                for name, value in current_state.items():
                    averaged = optimizer.ema_state[name]
                    if name in optimizer.ema_parameter_names:
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
                        averaged.lerp_(value.detach(), ema_rate)
                    else:
                        averaged.copy_(value)

            if step >= total_steps:
                for name, value in current_state.items():
                    value.copy_(optimizer.ema_state[name])
