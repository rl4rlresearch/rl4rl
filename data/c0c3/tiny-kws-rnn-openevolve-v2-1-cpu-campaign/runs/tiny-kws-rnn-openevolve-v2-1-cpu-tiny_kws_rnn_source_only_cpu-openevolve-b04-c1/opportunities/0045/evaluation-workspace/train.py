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
    """A causal GRU that performs one learned transition per adjacent frame pair."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(40, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(272, 7)

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
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
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
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        frame = self.input_norm(frame)

        if phase[0, 0].item() < 0.5:
            return (
                hidden,
                early_summary,
                late_summary,
                maximum,
                count,
                frame,
                torch.ones_like(phase),
            )

        paired = torch.cat((pending, frame), dim=1)
        output, hidden = self.gru(
            paired.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        early_weight = (count < 6.0).to(dtype=output.dtype)
        late_weight = 1.0 - early_weight
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            late_summary + late_weight * output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
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
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        if frames.shape[1] == 0:
            return state

        frames = self.input_norm(frames)
        paired_parts: list[torch.Tensor] = []
        position = 0

        if phase[0, 0].item() >= 0.5:
            paired_parts.append(
                torch.cat((pending.unsqueeze(1), frames[:, :1, :]), dim=2)
            )
            position = 1

        pair_count = (frames.shape[1] - position) // 2
        if pair_count > 0:
            end = position + 2 * pair_count
            paired_parts.append(
                torch.cat(
                    (
                        frames[:, position:end:2, :],
                        frames[:, position + 1:end:2, :],
                    ),
                    dim=2,
                )
            )
            position = end

        if paired_parts:
            paired = torch.cat(paired_parts, dim=1)
            outputs, hidden = self.gru(
                paired,
                hidden.transpose(0, 1).contiguous(),
            )
            sequence_maximum = outputs.amax(dim=1)
            maximum = torch.where(
                count > 0,
                torch.maximum(maximum, sequence_maximum),
                sequence_maximum,
            )
            positions = count.unsqueeze(1) + torch.arange(
                paired.shape[1],
                device=outputs.device,
                dtype=outputs.dtype,
            ).view(1, -1, 1)
            early_weights = (positions < 6.0).to(dtype=outputs.dtype)
            late_weights = 1.0 - early_weights
            early_summary = early_summary + (
                outputs * early_weights
            ).sum(dim=1)
            late_summary = late_summary + (
                outputs * late_weights
            ).sum(dim=1)
            count = count + paired.shape[1]
            hidden = hidden.transpose(0, 1)
            pending = torch.zeros_like(pending)
            phase = torch.zeros_like(phase)

        if position < frames.shape[1]:
            pending = frames[:, position, :]
            phase = torch.ones_like(phase)

        return (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
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
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        del pending, phase
        early_count = count.clamp(max=6.0).clamp_min(1.0)
        late_count = (count - 6.0).clamp_min(1.0)
        pooled = torch.cat(
            (
                early_summary / early_count,
                late_summary / late_count,
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
        relative_logits = self.classifier(pooled)
        logits = torch.cat(
            (
                relative_logits,
                torch.zeros_like(relative_logits[:, :1]),
            ),
            dim=1,
        )
        return logits - logits.mean(dim=1, keepdim=True)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(26, available_frames)
        steps -= steps % 2
        start = (available_frames - steps) // 2
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
