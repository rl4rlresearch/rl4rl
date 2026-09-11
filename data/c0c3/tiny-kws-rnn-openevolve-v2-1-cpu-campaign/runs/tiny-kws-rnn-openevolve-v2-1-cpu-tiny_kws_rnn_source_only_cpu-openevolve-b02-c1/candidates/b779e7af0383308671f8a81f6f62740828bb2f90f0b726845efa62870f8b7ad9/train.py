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
    """Hierarchical gated memory with frame-rate and five-frame state updates."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_cell = nn.GRUCell(35, 48)
        self.slow_cell = nn.GRUCell(48, 64)
        self.classifier = nn.Linear(160, 7)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fast = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return fast, slow, previous, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
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
    ]:
        fast, slow, previous, summary, count = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        delta_features = torch.cat(
            (
                delta[:, :10],
                delta[:, 10:12].mean(dim=1, keepdim=True),
                delta[:, 12:14].mean(dim=1, keepdim=True),
                delta[:, 14:16].mean(dim=1, keepdim=True),
                delta[:, 16:18].mean(dim=1, keepdim=True),
                delta[:, 18:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        features = torch.cat((normalized, delta_features), dim=1)
        fast = self.fast_cell(features, fast)

        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 5
        ) == 4
        if bool(slow_boundary.any()):
            updated = self.slow_cell(
                fast[slow_boundary], slow[slow_boundary]
            )
            next_slow = slow.clone()
            next_slow[slow_boundary] = updated
            slow = next_slow

        return fast, slow, normalized, summary + fast, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
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
    ]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        fast, slow, _previous, summary, count = state
        mean_fast = summary / count.clamp_min(1.0)
        readout = torch.cat((mean_fast, fast, slow), dim=1)
        coordinates = self.classifier(readout)
        coordinate_sum = coordinates.sum(dim=1, keepdim=True)
        scale = 1.0 / math.sqrt(8.0)
        correction = (scale * scale) / (1.0 - scale)
        return torch.cat(
            (
                coordinates - correction * coordinate_sum,
                scale * coordinate_sum,
            ),
            dim=1,
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 4))


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
