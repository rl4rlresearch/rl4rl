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
    """Unified gated memory over spectral levels and explicit local dynamics."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(36, 79, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(158, 7)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 79, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 79, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, previous, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, previous, summary, count = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        delta_features = torch.cat(
            (
                delta[:, :12],
                delta[:, 12:14].mean(dim=1, keepdim=True),
                delta[:, 14:16].mean(dim=1, keepdim=True),
                delta[:, 16:18].mean(dim=1, keepdim=True),
                delta[:, 18:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        features = torch.cat((normalized, delta_features), dim=1).unsqueeze(1)
        output, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        current = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            normalized,
            summary + current,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, previous, summary, count = state
        normalized = self.input_norm(frames)
        has_previous = (count > 0.0).to(dtype=normalized.dtype).unsqueeze(1)
        first_delta = (
            normalized[:, :1, :] - previous.unsqueeze(1)
        ) * has_previous
        remaining_deltas = normalized[:, 1:, :] - normalized[:, :-1, :]
        deltas = torch.cat((first_delta, remaining_deltas), dim=1)
        delta_features = torch.cat(
            (
                deltas[:, :, :12],
                deltas[:, :, 12:14].mean(dim=2, keepdim=True),
                deltas[:, :, 14:16].mean(dim=2, keepdim=True),
                deltas[:, :, 16:18].mean(dim=2, keepdim=True),
                deltas[:, :, 18:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
        features = torch.cat((normalized, delta_features), dim=2)
        outputs, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            normalized[:, -1, :],
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, _previous, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        readout = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
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
