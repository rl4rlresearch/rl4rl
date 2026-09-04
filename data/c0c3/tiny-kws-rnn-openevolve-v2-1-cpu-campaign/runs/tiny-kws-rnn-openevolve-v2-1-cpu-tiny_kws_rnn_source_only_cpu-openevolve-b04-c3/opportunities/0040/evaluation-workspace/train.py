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
    """A multirate causal hierarchy with pair-pooled contextual updates."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.acoustic_gru = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.context_norm = nn.LayerNorm(48)
        self.context_gru = nn.GRU(48, 120, num_layers=1, batch_first=True)
        self.mean_norm = nn.LayerNorm(168)
        self.terminal_norm = nn.LayerNorm(168)
        self.classifier = nn.Linear(336, 8)
        self.mean_aux = nn.Linear(168, 8)
        self.terminal_aux = nn.Linear(168, 8)
        self._aux_logits = None

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        acoustic_hidden = torch.zeros(
            batch_size, 1, 48, device=device, dtype=dtype
        )
        context_hidden = torch.zeros(
            batch_size, 1, 120, device=device, dtype=dtype
        )
        acoustic_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        context_summary = torch.zeros(
            batch_size, 120, device=device, dtype=dtype
        )
        block_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        frame_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        context_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            block_summary,
            frame_count,
            context_count,
        )

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            block_summary,
            frame_count,
            context_count,
        ) = state
        acoustic_output, next_acoustic_hidden = self.acoustic_gru(
            self.input_norm(frame).unsqueeze(1),
            acoustic_hidden.transpose(0, 1).contiguous(),
        )
        acoustic_output = acoustic_output[:, 0, :]
        acoustic_hidden = next_acoustic_hidden.transpose(0, 1)
        acoustic_summary = acoustic_summary + acoustic_output

        frame_index = int(frame_count[0, 0].detach().item())
        if frame_index % 2 == 0:
            if frame_index == 0:
                context_input = acoustic_output
            else:
                context_input = 0.5 * (
                    block_summary + acoustic_output
                )
            context_output, next_context_hidden = self.context_gru(
                self.context_norm(context_input).unsqueeze(1),
                context_hidden.transpose(0, 1).contiguous(),
            )
            context_hidden = next_context_hidden.transpose(0, 1)
            context_summary = context_summary + context_output[:, 0, :]
            context_count = context_count + 1.0
            block_summary = torch.zeros_like(block_summary)
        else:
            block_summary = acoustic_output

        return (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            block_summary,
            frame_count + 1.0,
            context_count,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        for frame_index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, frame_index, :], state)
        return state

    def classify(
        self, state: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            _block_summary,
            frame_count,
            context_count,
        ) = state
        mean = self.mean_norm(
            torch.cat(
                (
                    acoustic_summary / frame_count.clamp_min(1.0),
                    context_summary / context_count.clamp_min(1.0),
                ),
                dim=-1,
            )
        )
        terminal = self.terminal_norm(
            torch.cat(
                (
                    acoustic_hidden[:, 0, :],
                    context_hidden[:, 0, :],
                ),
                dim=-1,
            )
        )
        logits = self.classifier(torch.cat((mean, terminal), dim=-1))
        if self.training:
            self._aux_logits = (
                self.mean_aux(mean),
                self.terminal_aux(terminal),
            )
        else:
            self._aux_logits = None
        return logits

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
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
    main_loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
    aux_logits = getattr(model, "_aux_logits", None)
    if aux_logits is None:
        return main_loss
    mean_logits, terminal_logits = aux_logits
    mean_loss = F.cross_entropy(
        mean_logits, labels, label_smoothing=0.03
    )
    terminal_loss = F.cross_entropy(
        terminal_logits, labels, label_smoothing=0.03
    )
    return (main_loss + 0.1 * mean_loss + 0.1 * terminal_loss) / 1.2


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
