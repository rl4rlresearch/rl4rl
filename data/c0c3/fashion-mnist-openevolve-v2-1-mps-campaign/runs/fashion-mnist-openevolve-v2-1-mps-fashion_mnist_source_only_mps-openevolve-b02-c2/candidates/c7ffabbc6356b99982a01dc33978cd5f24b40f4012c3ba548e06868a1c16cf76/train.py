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
        self.stem_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(32)
        self.stem_refine_conv1 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.stem_refine_norm1 = nn.BatchNorm2d(32)
        self.stem_refine_conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.stem_refine_norm2 = nn.BatchNorm2d(32)

        self.block_conv1 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1, bias=False
        )
        self.block_norm1 = nn.BatchNorm2d(64)
        self.block_conv2 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=False
        )
        self.block_norm2 = nn.BatchNorm2d(64)
        self.projection = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.MaxPool2d(2)

        self.deep_conv1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.deep_norm1 = nn.BatchNorm2d(32)
        self.deep_conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.deep_norm2 = nn.BatchNorm2d(32)
        self.deep_conv3 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.deep_norm3 = nn.BatchNorm2d(64)

        self.post_deep_conv1 = nn.Conv2d(64, 24, kernel_size=1, bias=False)
        self.post_deep_norm1 = nn.BatchNorm2d(24)
        self.post_deep_conv2 = nn.Conv2d(
            24, 24, kernel_size=3, padding=1, bias=False
        )
        self.post_deep_norm2 = nn.BatchNorm2d(24)
        self.post_deep_conv3 = nn.Conv2d(24, 64, kernel_size=1, bias=False)
        self.post_deep_norm3 = nn.BatchNorm2d(64)

        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.spatial_attention = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = F.gelu(self.stem_norm(self.stem_conv(images)))
        residual = features
        features = F.gelu(
            self.stem_refine_norm1(self.stem_refine_conv1(features))
        )
        features = self.stem_refine_norm2(self.stem_refine_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = self.projection(features)
        features = F.gelu(self.block_norm1(self.block_conv1(features)))
        features = self.block_norm2(self.block_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = features
        features = F.gelu(self.deep_norm1(self.deep_conv1(features)))
        features = F.gelu(self.deep_norm2(self.deep_conv2(features)))
        features = self.deep_norm3(self.deep_conv3(features))
        features = F.gelu(features + residual)

        residual = features
        features = F.gelu(
            self.post_deep_norm1(self.post_deep_conv1(features))
        )
        features = F.gelu(
            self.post_deep_norm2(self.post_deep_conv2(features))
        )
        features = self.post_deep_norm3(self.post_deep_conv3(features))
        features = F.gelu(features + residual)

        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
        features = features * (2.0 * torch.sigmoid(channel_gate))

        spatial_summary = torch.cat(
            (
                features.mean(dim=1, keepdim=True),
                features.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_gate = self.spatial_attention(spatial_summary)
        features = features * (2.0 * torch.sigmoid(spatial_gate))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        top_two = log_probabilities.topk(2, dim=2).values
        margins = top_two[..., 0] - top_two[..., 1]
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
        fused_log_probabilities = torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
        return fused_log_probabilities / 0.85


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )


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
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
