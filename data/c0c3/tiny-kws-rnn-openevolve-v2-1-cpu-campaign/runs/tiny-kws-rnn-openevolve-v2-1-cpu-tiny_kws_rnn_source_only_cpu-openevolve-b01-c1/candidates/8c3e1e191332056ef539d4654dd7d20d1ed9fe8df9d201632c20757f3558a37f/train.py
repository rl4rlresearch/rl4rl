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


class SegmentMemoryNet(nn.Module):
    """A causal recurrent bank of ordered, nonlinear temporal summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.frame_encoder = nn.Sequential(
            nn.Linear(20, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.memory_norm = nn.LayerNorm(7 * 64 * 2)
        self.classifier = nn.Sequential(
            nn.Linear(7 * 64 * 2, 96),
            nn.SiLU(),
            nn.Linear(96, 8),
        )

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (batch_size, 7, 64)
        sums = torch.zeros(shape, device=device, dtype=dtype)
        maxima = torch.zeros(shape, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return sums, maxima, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sums, maxima, count = state
        encoded = self.frame_encoder(self.input_norm(frame))
        slot = (count.to(dtype=torch.long) // 4).clamp_max(6).squeeze(1)
        selector = F.one_hot(slot, num_classes=7).to(encoded.dtype).unsqueeze(-1)
        candidate = selector * encoded.unsqueeze(1)
        return sums + candidate, torch.maximum(maxima, candidate), count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sums, maxima, count = state
        encoded = self.frame_encoder(self.input_norm(frames))
        offsets = torch.arange(
            frames.shape[1], device=frames.device, dtype=torch.long
        ).unsqueeze(0)
        slots = (
            count.to(dtype=torch.long) + offsets
        ).div(4, rounding_mode="floor").clamp_max(6)
        selector = F.one_hot(slots, num_classes=7).to(encoded.dtype).unsqueeze(-1)
        candidates = selector * encoded.unsqueeze(2)
        return (
            sums + candidates.sum(dim=1),
            torch.maximum(maxima, candidates.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        sums, maxima, _count = state
        features = torch.cat((0.25 * sums, maxima), dim=-1).flatten(1)
        return self.classifier(self.memory_norm(features))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(4, available_frames))


def build_model() -> nn.Module:
    return SegmentMemoryNet()


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
