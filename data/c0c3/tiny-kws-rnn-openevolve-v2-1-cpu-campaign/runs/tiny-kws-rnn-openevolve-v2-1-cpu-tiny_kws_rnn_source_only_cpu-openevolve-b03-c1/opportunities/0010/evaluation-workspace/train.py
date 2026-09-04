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


class SharedGateRNN(nn.Module):
    """A wide causal recurrent cell with one dense hidden transform."""

    hidden_size = 160

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.candidate_input = nn.Linear(20, self.hidden_size)
        self.retain_input = nn.Linear(20, self.hidden_size)
        self.recurrent = nn.Linear(self.hidden_size, self.hidden_size)
        self.candidate_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(2 * self.hidden_size, 8)
        nn.init.constant_(self.retain_input.bias, 1.0)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros_like(hidden)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        frame = self.input_norm(frame)
        candidate = torch.tanh(
            self.candidate_norm(
                self.candidate_input(frame) + self.recurrent(hidden)
            )
        )
        retain = torch.sigmoid(self.retain_input(frame))
        hidden = retain * hidden + (1.0 - retain) * candidate
        return hidden, summary + hidden, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for frame in frames.unbind(dim=1):
            state = self.recurrent_step(frame, state)
        return state

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        temporal_mean = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((hidden, temporal_mean), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))


def build_model() -> nn.Module:
    return SharedGateRNN()


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
