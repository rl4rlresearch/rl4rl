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


class MultiScaleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = channels // 2
        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat(
            (self.local(features), self.context(features)),
            dim=1,
        )
        return F.gelu(features + self.fuse(mixed))


class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features + self.refine(features))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.early = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.mid_context = MultiScaleBlock(64)
        self.pool2 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.late_context = MultiScaleBlock(96)
        self.late_refinement = SpatialRefinement(96)
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.early(features))
        features = self.down1(self.pool1(features))
        features = self.mid_context(features)
        features = self.down2(self.pool2(features))
        features = self.late_context(features)
        features = self.late_refinement(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._predict(images)

        height, width = images.shape[-2:]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = []
        for row_offset, col_offset in (
            (1, 1),
            (0, 1),
            (2, 1),
            (1, 0),
            (1, 2),
        ):
            view = padded[
                :,
                :,
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
            views.append(view)
            views.append(view.flip(-1))

        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        offset_logits = view_logits.mean(dim=1)
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        agreement_uncertainty = view_agreement * (1.0 - view_agreement)
        calibration = 1.22775 * (
            0.92211476
            + 0.07788524 * view_agreement
            - 0.02 * agreement_uncertainty
        )
        return calibration.unsqueeze(1) * pooled_logits


def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5.0e-4,
        weight_decay=2e-4,
    )
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
    optimizer.tail_average_weight_sum = 0.0
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
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
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
    alternate_positions = (
        positions
        + torch.randint(1, 5, (batch,), device=images.device)
    ) % 5
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.25
    partner_positions = torch.where(
        cross_offset_mask,
        alternate_positions,
        positions,
    )

    row_choices = torch.tensor(
        (1, 0, 2, 1, 1),
        device=images.device,
    )
    col_choices = torch.tensor(
        (1, 1, 1, 0, 2),
        device=images.device,
    )

    def crop_views(selected_positions: torch.Tensor) -> torch.Tensor:
        row_offsets = row_choices[selected_positions].unsqueeze(1)
        col_offsets = col_choices[selected_positions].unsqueeze(1)
        rows = (
            row_offsets
            + torch.arange(height, device=images.device).unsqueeze(0)
        )
        row_index = rows[:, None, :, None].expand(
            batch,
            channels,
            height,
            padded.shape[-1],
        )
        cropped = padded.gather(2, row_index)
        cols = (
            col_offsets
            + torch.arange(width, device=images.device).unsqueeze(0)
        )
        col_index = cols[:, None, None, :].expand(
            batch,
            channels,
            height,
            width,
        )
        return cropped.gather(3, col_index)

    original_views = crop_views(positions)
    partner_views = crop_views(partner_positions)
    flip_mask = torch.rand(batch, device=images.device) < 0.5
    original_views = torch.where(
        flip_mask[:, None, None, None],
        original_views.flip(-1),
        original_views,
    )
    partner_views = torch.where(
        flip_mask[:, None, None, None],
        partner_views,
        partner_views.flip(-1),
    )

    paired_logits = model(
        torch.cat((original_views, partner_views), dim=0)
    )
    original_logits, partner_logits = paired_logits.chunk(2, dim=0)
    classification_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            partner_logits,
            labels,
            label_smoothing=0.02,
        )
    )

    original_log_probabilities = F.log_softmax(original_logits, dim=-1)
    partner_log_probabilities = F.log_softmax(partner_logits, dim=-1)
    consistency_loss = 0.5 * (
        F.kl_div(
            original_log_probabilities,
            partner_log_probabilities.exp(),
            reduction="none",
        ).sum(dim=-1)
        + F.kl_div(
            partner_log_probabilities,
            original_log_probabilities.exp(),
            reduction="none",
        ).sum(dim=-1)
    )
    consistency_weights = torch.where(
        cross_offset_mask,
        consistency_loss.new_tensor(0.075),
        consistency_loss.new_tensor(0.05),
    )
    return classification_loss + (
        consistency_weights * consistency_loss
    ).mean()


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.2 + 0.8 * progress / warmup_fraction
    else:
        decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        multiplier = 0.04 + 0.96 * 0.5 * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    tail_index = step + 2 - optimizer.tail_average_start
    tail_length = total_steps - optimizer.tail_average_start + 1
    if (
        step + 1 >= optimizer.tail_average_start
        and (tail_index == 1 or tail_index % 2 == 0)
    ):
        recency_power = 1.0
        if tail_index == 1:
            sample_weight = 1.0
        elif tail_index == 2:
            sample_weight = (
                tail_index ** recency_power
                + 0.5 * (tail_index + 1) ** recency_power
            )
        elif tail_index == tail_length:
            sample_weight = (
                tail_index ** recency_power
                + 0.5 * (tail_index - 1) ** recency_power
            )
        else:
            sample_weight = (
                tail_index ** recency_power
                + 0.5 * (tail_index - 1) ** recency_power
                + 0.5 * (tail_index + 1) ** recency_power
            )

        optimizer.tail_average_weight_sum += sample_weight
        average_weight = sample_weight / optimizer.tail_average_weight_sum
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.tail_average_parameters,
                parameters,
            ):
                average.lerp_(parameter, average_weight)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer.tail_average_parameters,
                ):
                    parameter.copy_(average)
