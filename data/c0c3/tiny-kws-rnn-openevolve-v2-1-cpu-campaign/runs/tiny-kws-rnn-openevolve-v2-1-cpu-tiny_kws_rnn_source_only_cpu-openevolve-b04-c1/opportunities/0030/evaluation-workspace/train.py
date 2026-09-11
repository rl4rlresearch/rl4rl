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
    """A causal hierarchy with distinct phonetic and command timescales."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_gru = nn.GRU(20, 42, num_layers=1, batch_first=True)
        self.slow_gru = nn.GRU(42, 52, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(282, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        fast_hidden = torch.zeros(batch_size, 1, 42, device=device, dtype=dtype)
        slow_hidden = torch.zeros(batch_size, 1, 52, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 42, device=device, dtype=dtype)
        fast_maximum = torch.zeros(batch_size, 42, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 52, device=device, dtype=dtype)
        slow_maximum = torch.zeros(batch_size, 52, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        slow_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            fast_hidden,
            slow_hidden,
            fast_summary,
            fast_maximum,
            slow_summary,
            slow_maximum,
            count,
            slow_count,
        )

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            fast_maximum,
            slow_summary,
            slow_maximum,
            count,
            slow_count,
        ) = state

        fast_output, fast_hidden = self.fast_gru(
            self.input_norm(frame).unsqueeze(1),
            fast_hidden.transpose(0, 1).contiguous(),
        )
        fast_output = fast_output[:, 0, :]
        fast_maximum = torch.where(
            count > 0,
            torch.maximum(fast_maximum, fast_output),
            fast_output,
        )
        count = count + 1.0

        if int(count[0, 0].item()) % 4 == 0:
            slow_output, slow_hidden = self.slow_gru(
                fast_output.unsqueeze(1),
                slow_hidden.transpose(0, 1).contiguous(),
            )
            slow_output = slow_output[:, 0, :]
            slow_maximum = torch.where(
                slow_count > 0,
                torch.maximum(slow_maximum, slow_output),
                slow_output,
            )
            slow_summary = slow_summary + slow_output
            slow_count = slow_count + 1.0

        return (
            fast_hidden.transpose(0, 1),
            slow_hidden.transpose(0, 1),
            fast_summary + fast_output,
            fast_maximum,
            slow_summary,
            slow_maximum,
            count,
            slow_count,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            fast_maximum,
            slow_summary,
            slow_maximum,
            count,
            slow_count,
        ) = state

        fast_outputs, fast_hidden = self.fast_gru(
            self.input_norm(frames),
            fast_hidden.transpose(0, 1).contiguous(),
        )
        sequence_maximum = fast_outputs.amax(dim=1)
        fast_maximum = torch.where(
            count > 0,
            torch.maximum(fast_maximum, sequence_maximum),
            sequence_maximum,
        )

        start_step = int(count[0, 0].item())
        slow_positions = [
            index
            for index in range(frames.shape[1])
            if (start_step + index + 1) % 4 == 0
        ]
        if slow_positions:
            slow_outputs, slow_hidden = self.slow_gru(
                fast_outputs[:, slow_positions, :],
                slow_hidden.transpose(0, 1).contiguous(),
            )
            sequence_slow_maximum = slow_outputs.amax(dim=1)
            slow_maximum = torch.where(
                slow_count > 0,
                torch.maximum(slow_maximum, sequence_slow_maximum),
                sequence_slow_maximum,
            )
            slow_summary = slow_summary + slow_outputs.sum(dim=1)
            slow_count = slow_count + len(slow_positions)

        return (
            fast_hidden.transpose(0, 1),
            slow_hidden.transpose(0, 1),
            fast_summary + fast_outputs.sum(dim=1),
            fast_maximum,
            slow_summary,
            slow_maximum,
            count + frames.shape[1],
            slow_count,
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            fast_maximum,
            slow_summary,
            slow_maximum,
            count,
            slow_count,
        ) = state
        pooled = torch.cat(
            (
                fast_summary / count.clamp_min(1.0),
                fast_maximum,
                fast_hidden[:, 0, :],
                slow_summary / slow_count.clamp_min(1.0),
                slow_maximum,
                slow_hidden[:, 0, :],
            ),
            dim=1,
        )
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        start = (available_frames - steps + 1) // 2
        return list(range(start, start + steps))


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
