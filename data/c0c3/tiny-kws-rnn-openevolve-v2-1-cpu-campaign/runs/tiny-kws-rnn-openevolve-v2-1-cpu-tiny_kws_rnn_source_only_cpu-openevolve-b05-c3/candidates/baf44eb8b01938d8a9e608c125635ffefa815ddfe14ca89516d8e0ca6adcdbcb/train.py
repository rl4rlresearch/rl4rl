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

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A clocked hierarchy with separate acoustic and command timescales."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_cell = nn.GRUCell(20, 80)
        self.slow_norm = nn.LayerNorm(80)
        self.slow_cell = nn.GRUCell(80, 80)
        self.classifier = nn.Linear(240, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fast = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        slow_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            fast,
            slow,
            block_sum,
            fast_summary,
            slow_summary,
            count,
            slow_count,
        )

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            fast,
            slow,
            block_sum,
            fast_summary,
            slow_summary,
            count,
            slow_count,
        ) = state

        fast = self.fast_cell(self.input_norm(frame), fast)
        block_sum = block_sum + fast
        fast_summary = fast_summary + fast
        count = count + 1.0

        if int(count[0, 0].detach().item()) % 4 == 0:
            slow_input = self.slow_norm(block_sum * 0.25)
            slow = self.slow_cell(slow_input, slow)
            slow_summary = slow_summary + slow
            slow_count = slow_count + 1.0
            block_sum = torch.zeros_like(block_sum)

        return (
            fast,
            slow,
            block_sum,
            fast_summary,
            slow_summary,
            count,
            slow_count,
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            _fast,
            slow,
            _block_sum,
            fast_summary,
            slow_summary,
            count,
            slow_count,
        ) = state
        fast_mean = fast_summary / count.clamp_min(1.0)
        slow_mean = slow_summary / slow_count.clamp_min(1.0)
        return self.classifier(torch.cat((fast_mean, slow_mean, slow), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
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
