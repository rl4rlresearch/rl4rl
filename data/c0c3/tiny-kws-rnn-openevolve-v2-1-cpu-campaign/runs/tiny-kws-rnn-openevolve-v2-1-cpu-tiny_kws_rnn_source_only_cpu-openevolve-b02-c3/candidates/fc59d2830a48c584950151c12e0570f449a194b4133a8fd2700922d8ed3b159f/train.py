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
    """A wide gated state controlled through a compact recurrent bottleneck."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_gates = nn.Linear(40, 252)
        self.state_encoder = nn.Linear(84, 28, bias=False)
        self.state_gates = nn.Linear(28, 252)
        self.classifier = nn.Linear(168, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        return hidden, summary, count, previous

    def _update_hidden(
        self, input_drive: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        controller = torch.tanh(self.state_encoder(hidden))
        state_drive = self.state_gates(controller)
        input_reset, input_update, input_candidate = input_drive.chunk(3, dim=1)
        state_reset, state_update, state_candidate = state_drive.chunk(3, dim=1)
        reset = torch.sigmoid(input_reset + state_reset)
        update = torch.sigmoid(input_update + state_update)
        candidate = torch.tanh(input_candidate + reset * state_candidate)
        return update * hidden + (1.0 - update) * candidate

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1)
        hidden = self._update_hidden(self.input_gates(features), hidden)
        return hidden, summary + hidden, count + 1.0, normalized

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frames)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        first_delta = (normalized[:, 0, :] - previous) * has_previous
        deltas = torch.cat(
            (
                first_delta.unsqueeze(1),
                normalized[:, 1:, :] - normalized[:, :-1, :],
            ),
            dim=1,
        )
        features = torch.cat((normalized, deltas), dim=2)
        input_drives = self.input_gates(features)
        for index in range(frames.shape[1]):
            hidden = self._update_hidden(input_drives[:, index, :], hidden)
            summary = summary + hidden
        return (
            hidden,
            summary,
            count + frames.shape[1],
            normalized[:, -1, :],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, count, previous = state
        del previous
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(27, available_frames)
        start = max(available_frames - 28, 0)
        window_frames = available_frames - start
        if window_frames <= target_steps:
            return list(range(start, available_frames))
        return [
            start + round(index * (window_frames - 1) / (target_steps - 1))
            for index in range(target_steps)
        ]


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
