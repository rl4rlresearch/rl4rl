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


class CoupledForgetCell(nn.Module):
    """Single-gate recurrent cell with a coupled input gate."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.affine = nn.Linear(input_size + hidden_size, 2 * hidden_size)

    def forward(
        self, frame: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        forget_logits, candidate_logits = self.affine(
            torch.cat((frame, hidden), dim=1)
        ).chunk(2, dim=1)
        forget = torch.sigmoid(forget_logits)
        candidate = torch.tanh(candidate_logits)
        return forget * hidden + (1.0 - forget) * candidate


class KeywordGRU(nn.Module):
    """Full-GRU anchor with a cheaper coupled-gate auxiliary branch."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.cell_b = CoupledForgetCell(20, 59)
        self.classifier = nn.Linear(119, 7)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a, hidden_b, summary, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        output_a, next_hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        next_hidden_b = self.cell_b(normalized[:, 0, :], hidden_b[:, 0, :])
        output = torch.cat((output_a[:, 0, :], next_hidden_b), dim=1)
        return (
            next_hidden_a.transpose(0, 1),
            next_hidden_b.unsqueeze(1),
            summary + output,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a, hidden_b, summary, count = state
        normalized = self.input_norm(frames)
        outputs_a, next_hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        current_b = hidden_b[:, 0, :]
        outputs_b = []
        for normalized_frame in normalized.unbind(dim=1):
            current_b = self.cell_b(normalized_frame, current_b)
            outputs_b.append(current_b)
        stacked_b = torch.stack(outputs_b, dim=1)
        outputs = torch.cat((outputs_a, stacked_b), dim=2)
        return (
            next_hidden_a.transpose(0, 1),
            current_b.unsqueeze(1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, summary, count = state
        coordinates = self.classifier(summary / count.clamp_min(1.0))
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
        return list(range(3, available_frames))


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
