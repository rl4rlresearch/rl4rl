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
    """Parallel gated recurrent subspaces with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.branches = nn.ModuleList(
            [
                nn.GRU(20, 52, num_layers=1, batch_first=True),
                nn.GRU(20, 52, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(208, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 2, 52, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        outputs = []
        next_hidden = []
        for index, branch in enumerate(self.branches):
            output, branch_hidden = branch(
                normalized,
                hidden[:, index, :].unsqueeze(0).contiguous(),
            )
            outputs.append(output[:, 0, :])
            next_hidden.append(branch_hidden[0])
        combined = torch.cat(outputs, dim=1)
        return (
            torch.stack(next_hidden, dim=1),
            summary + combined,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        normalized = self.input_norm(frames)
        outputs = []
        next_hidden = []
        for index, branch in enumerate(self.branches):
            branch_outputs, branch_hidden = branch(
                normalized,
                hidden[:, index, :].unsqueeze(0).contiguous(),
            )
            outputs.append(branch_outputs)
            next_hidden.append(branch_hidden[0])
        combined = torch.cat(outputs, dim=2)
        return (
            torch.stack(next_hidden, dim=1),
            summary + combined.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden.reshape(hidden.shape[0], 104)
        return self.classifier(torch.cat((mean_output, final_output), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 30, 0)
        return list(range(start, available_frames))


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
