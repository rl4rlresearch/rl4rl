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
GRAD_CLIP_NORM = 1.0


class SpatialRelationBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(96)
        self.attention = nn.MultiheadAttention(
            96, num_heads=4, dropout=0.10, batch_first=True
        )
        self.norm2 = nn.LayerNorm(96)
        self.mlp = nn.Sequential(
            nn.Linear(96, 144),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(144, 96),
            nn.Dropout(0.15),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class RelationalClassHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_positions = nn.Parameter(torch.empty(1, 9, 96))
        self.class_tokens = nn.Parameter(torch.empty(1, 10, 96))
        self.spatial_relations = SpatialRelationBlock()
        self.class_norm = nn.LayerNorm(96)
        self.memory_norm = nn.LayerNorm(96)
        self.class_attention = nn.MultiheadAttention(
            96, num_heads=4, dropout=0.10, batch_first=True
        )
        self.output_norm = nn.LayerNorm(96)
        self.class_mlp = nn.Sequential(
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 96),
            nn.Dropout(0.15),
        )
        self.class_readout = nn.Parameter(torch.empty(10, 96))
        self.class_bias = nn.Parameter(torch.zeros(10))

        nn.init.normal_(self.spatial_positions, std=0.02)
        nn.init.normal_(self.class_tokens, std=0.02)
        nn.init.normal_(self.class_readout, std=96 ** -0.5)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        spatial = feature_maps.flatten(2).transpose(1, 2)
        spatial = self.spatial_relations(spatial + self.spatial_positions)

        class_states = self.class_tokens.expand(feature_maps.shape[0], -1, -1)
        attended = self.class_attention(
            self.class_norm(class_states),
            self.memory_norm(spatial),
            self.memory_norm(spatial),
            need_weights=False,
        )[0]
        class_states = class_states + attended
        class_states = class_states + self.class_mlp(
            self.output_norm(class_states)
        )
        return (
            class_states * self.class_readout.unsqueeze(0)
        ).sum(dim=-1) + self.class_bias


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = RelationalClassHead()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return ensemble - math.log(12.0)


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)


def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    batch = images.shape[0]
    choices = torch.randint(0, 6, (batch,), device=images.device)
    offset_y = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_x = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_y = torch.where(choices == 2, 0, offset_y)
    offset_y = torch.where(choices == 3, 2, offset_y)
    offset_x = torch.where(choices == 4, 0, offset_x)
    offset_x = torch.where(choices == 5, 2, offset_x)
    windows = F.pad(images, (1, 1, 1, 1)).unfold(2, 28, 1).unfold(3, 28, 1)
    indices = torch.arange(batch, device=images.device)
    images = windows[indices, :, offset_y, offset_x, :, :]
    flip = torch.rand(batch, 1, 1, 1, device=images.device) < 0.5
    images = torch.where(flip, images.flip(-1), images)
    return images, labels


def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup = 0.08
    if progress < warmup:
        learning_rate = 8e-4 + (3.0e-3 - 8e-4) * progress / warmup
    else:
        decay = (progress - warmup) / (1.0 - warmup)
        multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * decay))
        learning_rate = 3.0e-3 * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    update = step + 1
    in_tail = update >= int(0.8 * total_steps)
    sample_tail = in_tail and (update % 8 == 0 or update == total_steps)
    if sample_tail:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_average"):
                optimizer._tail_average = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._tail_average_count = 1
            else:
                optimizer._tail_average_count += 1
                weight = 1.0 / optimizer._tail_average_count
                for average, parameter in zip(
                    optimizer._tail_average, parameters
                ):
                    average.lerp_(parameter.detach(), weight)

            if update == total_steps:
                for parameter, average in zip(
                    parameters, optimizer._tail_average
                ):
                    parameter.lerp_(average, 0.5)
