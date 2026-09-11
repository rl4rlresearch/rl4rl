"""Editable recurrent keyword-spotting research program.

The protected evaluator owns the audio, speaker-disjoint splits, log-mel
frontend, training exposure, recurrent execution loop, and exact MAC counter.
This file owns the recurrent model and trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 96
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)
        self.register_buffer(
            "classifier_keep_indices",
            torch.arange(114, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "classifier_dropped_indices",
            torch.zeros(2, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "classifier_dropped_weight",
            torch.zeros(7, 2),
            persistent=False,
        )
        self.register_buffer(
            "classifier_dropped_mean",
            torch.zeros(2),
            persistent=False,
        )
        self.register_buffer(
            "classifier_feature_sum",
            torch.zeros(114),
            persistent=False,
        )
        self.register_buffer(
            "classifier_feature_square_sum",
            torch.zeros(114),
            persistent=False,
        )
        self.register_buffer(
            "classifier_feature_count",
            torch.zeros(()),
            persistent=False,
        )

    def train(self, mode: bool = True) -> KeywordGRU:
        if mode and self.classifier.out_features == 7:
            reference_classifier = self.classifier
            expanded = nn.Linear(114, 8, bias=True).to(
                device=reference_classifier.weight.device,
                dtype=reference_classifier.weight.dtype,
            )
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.bias.zero_()
                expanded.weight[:7].index_copy_(
                    1,
                    self.classifier_keep_indices,
                    reference_classifier.weight,
                )
                expanded.weight[:7].index_copy_(
                    1,
                    self.classifier_dropped_indices,
                    self.classifier_dropped_weight,
                )
                bias_correction = (
                    self.classifier_dropped_weight
                    * self.classifier_dropped_mean.unsqueeze(0)
                ).sum(dim=1)
                expanded.bias[:7].copy_(
                    reference_classifier.bias - bias_correction
                )
            self.classifier = expanded
        elif not mode and self.classifier.out_features == 8:
            full_classifier = self.classifier
            relative_weight = (
                full_classifier.weight[:7] - full_classifier.weight[7:8]
            )
            centered_weight = full_classifier.weight - full_classifier.weight.mean(
                dim=0, keepdim=True
            )
            feature_count = self.classifier_feature_count.clamp_min(1.0)
            feature_mean = self.classifier_feature_sum / feature_count
            feature_variance = (
                self.classifier_feature_square_sum / feature_count
                - feature_mean.square()
            ).clamp_min(0.0)
            feature_variance = torch.where(
                self.classifier_feature_count > 0,
                feature_variance,
                torch.ones_like(feature_variance),
            )
            sensitivity = (
                centered_weight.square().sum(dim=0) * feature_variance
            )
            dropped_indices = sensitivity.topk(2, largest=False).indices
            all_indices = torch.arange(
                full_classifier.in_features,
                device=full_classifier.weight.device,
            )
            keep_mask = torch.ones_like(all_indices, dtype=torch.bool)
            keep_mask[dropped_indices] = False
            keep_indices = all_indices[keep_mask]
            dropped_mean = feature_mean.index_select(0, dropped_indices)
            dropped_weight = relative_weight.index_select(
                1, dropped_indices
            )
            compressed = nn.Linear(
                full_classifier.in_features - 2, 7, bias=True
            ).to(
                device=full_classifier.weight.device,
                dtype=full_classifier.weight.dtype,
            )
            with torch.no_grad():
                compressed.weight.copy_(
                    relative_weight.index_select(1, keep_indices)
                )
                bias_correction = (
                    dropped_weight * dropped_mean.unsqueeze(0)
                ).sum(dim=1)
                compressed.bias.copy_(
                    full_classifier.bias[:7]
                    - full_classifier.bias[7]
                    + bias_correction
                )
                self.classifier_keep_indices = keep_indices
                self.classifier_dropped_indices = dropped_indices
                self.classifier_dropped_weight = dropped_weight.clone()
                self.classifier_dropped_mean = dropped_mean.clone()
            self.classifier = compressed
        return super().train(mode)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output[:, :-2], hidden[:, 0, :]), dim=-1
        )
        if self.training:
            with torch.no_grad():
                detached_features = features.detach().to(
                    dtype=self.classifier_feature_sum.dtype
                )
                self.classifier_feature_sum.add_(
                    detached_features.sum(dim=0)
                )
                self.classifier_feature_square_sum.add_(
                    detached_features.square().sum(dim=0)
                )
                self.classifier_feature_count.add_(features.shape[0])
        if features.shape[-1] != self.classifier.in_features:
            features = features.index_select(
                -1, self.classifier_keep_indices
            )
        logits = self.classifier(features)
        if logits.shape[-1] == 7:
            logits = torch.cat((logits, logits.new_zeros(logits.shape[0], 1)), dim=-1)
        return logits

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames >= 8:
            full_window = list(range(2, available_frames - 2))
            schedule = full_window[1:-6] + full_window[-3::2]
            if len(schedule) > 2:
                return schedule[:1] + schedule[2:4] + schedule[5:]
            return schedule
        return list(range(available_frames))


def build_model() -> nn.Module:
    return KeywordGRU()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)


def prepare_training_batch(
    frames: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    if torch.rand(()) < 0.8:
        frames = frames + 0.025 * torch.randn_like(frames)
    return frames, labels


def training_loss(
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del model, step, total_steps
    return F.cross_entropy(logits, labels, label_smoothing=0.03)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    multiplier = 0.05 + 0.95 * 0.5 * (
        1.0 + math.cos(math.pi * step / max(total_steps, 1))
    )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
