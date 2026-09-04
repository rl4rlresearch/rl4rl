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


class KeywordSingleGate(nn.Module):
    """A full-rank single-gate recurrence with multi-view temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 81
        self.input_norm = nn.LayerNorm(20)
        self.transition = nn.Linear(20 + self.hidden_size, 2 * self.hidden_size)
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
        with torch.no_grad():
            self.transition.bias[: self.hidden_size].fill_(1.0)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros_like(hidden)
        peak = torch.zeros_like(hidden)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, peak, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        update = self.transition(
            torch.cat((self.input_norm(frame), hidden), dim=-1)
        )
        retention_logits, proposal_logits = update.chunk(2, dim=-1)
        retention = torch.sigmoid(retention_logits)
        proposal = torch.tanh(self.proposal_norm(proposal_logits))
        hidden = retention * hidden + (1.0 - retention) * proposal
        return (
            hidden,
            summary + hidden,
            torch.maximum(peak, hidden),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, peak, hidden), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        base_steps = min(27, available_frames)
        indices = [
            i * (available_frames - 1) // (base_steps - 1)
            for i in range(base_steps)
        ]
        if len(indices) > 26:
            del indices[1:4]
        return indices


def build_model() -> nn.Module:
    return KeywordSingleGate()


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
