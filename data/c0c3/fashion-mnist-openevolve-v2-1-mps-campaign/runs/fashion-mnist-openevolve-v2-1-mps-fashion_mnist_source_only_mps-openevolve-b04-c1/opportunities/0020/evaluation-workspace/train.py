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

        token_width = 80
        self.token_projection = nn.Sequential(
            nn.Conv2d(64, token_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(token_width),
            nn.GELU(),
        )
        self.image_positions = nn.Parameter(
            torch.empty(1, 7 * 7, token_width)
        )
        self.class_tokens = nn.Parameter(
            torch.empty(1, 10, token_width)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_width,
            nhead=4,
            dim_feedforward=2 * token_width,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )
        self.output_norm = nn.LayerNorm(token_width)
        self.classifier_weight = nn.Parameter(
            torch.empty(10, token_width)
        )
        self.classifier_bias = nn.Parameter(torch.zeros(10))

        nn.init.trunc_normal_(self.image_positions, std=0.02)
        nn.init.trunc_normal_(self.class_tokens, std=0.02)
        nn.init.trunc_normal_(self.classifier_weight, std=0.02)

    def _forward_view(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = self.pool(features)
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        features = self.pool(features)

        image_tokens = self.token_projection(features)
        image_tokens = image_tokens.flatten(2).transpose(1, 2)
        image_tokens = image_tokens + self.image_positions
        class_tokens = self.class_tokens.expand(
            images.shape[0], -1, -1
        )
        tokens = torch.cat((class_tokens, image_tokens), dim=1)
        tokens = self.global_encoder(tokens)
        class_features = self.output_norm(tokens[:, :10])
        return (
            class_features * self.classifier_weight.unsqueeze(0)
        ).sum(dim=-1) + self.classifier_bias

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_view(images)
        if self.training:
            return logits
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)


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
    del step, total_steps
    batch_size = images.shape[0]
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_logits = model(paired_images)
    logits = 0.5 * (
        paired_logits[:batch_size] + paired_logits[batch_size:]
    )
    return F.cross_entropy(logits, labels, label_smoothing=0.02)


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
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)

            if step >= total_steps:
                for name, value in current_state.items():
                    value.copy_(optimizer.ema_state[name])
