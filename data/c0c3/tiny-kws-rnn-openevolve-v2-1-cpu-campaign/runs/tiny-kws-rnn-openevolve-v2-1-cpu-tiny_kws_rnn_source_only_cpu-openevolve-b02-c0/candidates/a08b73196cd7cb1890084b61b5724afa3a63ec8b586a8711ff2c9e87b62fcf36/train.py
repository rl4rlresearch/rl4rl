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

BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
        self.exit_readout_start = 20

    def _input_features(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(frames)
        high_bands = normalized[..., 16:].reshape(
            *normalized.shape[:-1], 2, 2
        ).mean(dim=-1)
        return torch.cat((normalized[..., :16], high_bands), dim=-1)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self._input_features(frame).unsqueeze(1),
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
            self._input_features(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        mean_summary = summary / count.clamp_min(1.0)
        if bool((count[:, 0] < float(self.exit_readout_start)).all()):
            return mean_summary[:, :8]
        return self.classifier(mean_summary)

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        executed = count[:, 0]
        eligible = (
            (executed >= float(self.exit_readout_start))
            & (executed >= 2.0)
            & (executed < float(total_steps))
        )
        if not bool(eligible.any()):
            return torch.zeros_like(eligible)

        winner = logits.argmax(dim=-1)
        margins = logits.gather(1, winner.unsqueeze(1)) - logits

        weight = self.classifier.weight
        winner_weight = weight.index_select(0, winner)
        weight_delta = winner_weight.unsqueeze(1) - weight.unsqueeze(0)
        future_sensitivity = 1.0001 * weight_delta.abs().sum(dim=-1)

        bias = self.classifier.bias
        winner_bias = bias.index_select(0, winner)
        bias_delta = winner_bias.unsqueeze(1) - bias.unsqueeze(0)

        remaining = float(total_steps) - executed
        lower_final_margin = (
            executed.unsqueeze(1) * margins
            + remaining.unsqueeze(1) * (bias_delta - future_sensitivity)
        )
        class_ids = torch.arange(
            logits.shape[1], device=logits.device
        ).unsqueeze(0)
        competitors_safe = (
            (lower_final_margin > 1.0e-5)
            | (class_ids == winner.unsqueeze(1))
        )
        return eligible & competitors_safe.all(dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))


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
