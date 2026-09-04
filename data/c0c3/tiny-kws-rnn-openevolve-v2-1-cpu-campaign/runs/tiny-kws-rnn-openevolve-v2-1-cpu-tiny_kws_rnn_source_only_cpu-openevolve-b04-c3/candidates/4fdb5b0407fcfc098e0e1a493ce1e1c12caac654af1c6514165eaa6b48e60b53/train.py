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
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.mean_classifier = nn.Linear(125, 8)
        self.terminal_classifier = nn.Linear(125, 8)
        self.early_probe = nn.Linear(16, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 125, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 125, device=device, dtype=dtype)
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
        terminal = hidden[:, 0, :]
        probe_logits = self.early_probe(terminal[:, :16])
        self._probe_logits = probe_logits

        if not self.training and count.max().item() < 23.0:
            return probe_logits

        mean_logits = self.mean_classifier(
            summary / count.clamp_min(1.0)
        )
        terminal_logits = self.terminal_classifier(terminal)
        self._mean_logits = mean_logits
        self._terminal_logits = terminal_logits
        return mean_logits + terminal_logits

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        if total_steps <= 23:
            return torch.zeros(
                logits.shape[0], device=logits.device, dtype=torch.bool
            )

        prediction = logits.argmax(dim=-1)
        mean_probabilities = torch.softmax(self._mean_logits, dim=-1)
        terminal_probabilities = torch.softmax(
            self._terminal_logits, dim=-1
        )
        combined_confidence = torch.softmax(logits, dim=-1).amax(dim=-1)
        agreement = (
            (self._mean_logits.argmax(dim=-1) == prediction)
            & (self._terminal_logits.argmax(dim=-1) == prediction)
        )
        return (
            (count[:, 0] >= 23.0)
            & (count[:, 0] < float(total_steps))
            & agreement
            & (combined_confidence >= 0.995)
            & (mean_probabilities.amax(dim=-1) >= 0.90)
            & (terminal_probabilities.amax(dim=-1) >= 0.90)
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
        return schedule


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
    del step, total_steps
    main_loss = F.cross_entropy(
        logits, labels, label_smoothing=0.03
    )
    view_loss = 0.5 * (
        F.cross_entropy(
            model._mean_logits, labels, label_smoothing=0.03
        )
        + F.cross_entropy(
            model._terminal_logits, labels, label_smoothing=0.03
        )
    )
    probe_loss = F.cross_entropy(
        model._probe_logits, labels, label_smoothing=0.03
    )
    return main_loss + 0.10 * view_loss + 0.02 * probe_loss


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
