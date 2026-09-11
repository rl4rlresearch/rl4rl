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
    """A pyramidal hierarchy with a full-rate lower and half-rate upper stage."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(19)
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_gru = nn.GRU(65, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)

    @staticmethod
    def _fold_bands(frames: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                frames[..., :18],
                frames[..., 18:20].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        lower_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        upper_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending_acoustic = torch.zeros(
            batch_size, 16, device=device, dtype=dtype
        )
        return (
            lower_hidden,
            upper_hidden,
            lower_summary,
            upper_summary,
            count,
            upper_count,
            pending_acoustic,
        )

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            lower_hidden,
            upper_hidden,
            lower_summary,
            upper_summary,
            count,
            upper_count,
            pending_acoustic,
        ) = state
        normalized = self.input_norm(self._fold_bands(frame))
        lower_output, lower_hidden_time = self.lower_gru(
            normalized.unsqueeze(1),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        lower_output = lower_output[:, 0, :]
        lower_hidden = lower_hidden_time.transpose(0, 1)
        upper_acoustic = torch.cat(
            (
                normalized[..., :14],
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )

        if int(count[0, 0].item()) % 2 == 0:
            pending_acoustic = upper_acoustic
        else:
            pair_acoustic = 0.5 * (pending_acoustic + upper_acoustic)
            upper_input = torch.cat((pair_acoustic, lower_output), dim=-1)
            upper_output, upper_hidden_time = self.upper_gru(
                upper_input.unsqueeze(1),
                upper_hidden.transpose(0, 1).contiguous(),
            )
            upper_hidden = upper_hidden_time.transpose(0, 1)
            upper_summary = upper_summary + upper_output[:, 0, :]
            upper_count = upper_count + 1.0
            pending_acoustic = torch.zeros_like(pending_acoustic)

        return (
            lower_hidden,
            upper_hidden,
            lower_summary + lower_output,
            upper_summary,
            count + 1.0,
            upper_count,
            pending_acoustic,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            lower_hidden,
            upper_hidden,
            lower_summary,
            upper_summary,
            count,
            upper_count,
            pending_acoustic,
        ) = state
        normalized = self.input_norm(self._fold_bands(frames))
        lower_outputs, lower_hidden_time = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        lower_hidden = lower_hidden_time.transpose(0, 1)
        upper_acoustic = torch.cat(
            (
                normalized[..., :14],
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )

        total_steps = frames.shape[1]
        if int(count[0, 0].item()) % 2 == 0:
            paired_steps = (total_steps // 2) * 2
            pair_acoustic = 0.5 * (
                upper_acoustic[:, 0:paired_steps:2, :]
                + upper_acoustic[:, 1:paired_steps:2, :]
            )
            pair_lower = lower_outputs[:, 1:paired_steps:2, :]
            if total_steps % 2:
                pending_acoustic = upper_acoustic[:, -1, :]
            else:
                pending_acoustic = torch.zeros_like(pending_acoustic)
        else:
            first_acoustic = 0.5 * (
                pending_acoustic + upper_acoustic[:, 0, :]
            )
            pair_acoustic = first_acoustic.unsqueeze(1)
            pair_lower = lower_outputs[:, 0:1, :]
            paired_steps = ((total_steps - 1) // 2) * 2
            if paired_steps:
                remaining_acoustic = 0.5 * (
                    upper_acoustic[:, 1 : 1 + paired_steps : 2, :]
                    + upper_acoustic[:, 2 : 1 + paired_steps : 2, :]
                )
                pair_acoustic = torch.cat(
                    (pair_acoustic, remaining_acoustic), dim=1
                )
                pair_lower = torch.cat(
                    (
                        pair_lower,
                        lower_outputs[:, 2 : 1 + paired_steps : 2, :],
                    ),
                    dim=1,
                )
            if (total_steps - 1) % 2:
                pending_acoustic = upper_acoustic[:, -1, :]
            else:
                pending_acoustic = torch.zeros_like(pending_acoustic)

        if pair_acoustic.shape[1]:
            upper_inputs = torch.cat((pair_acoustic, pair_lower), dim=-1)
            upper_outputs, upper_hidden_time = self.upper_gru(
                upper_inputs,
                upper_hidden.transpose(0, 1).contiguous(),
            )
            upper_hidden = upper_hidden_time.transpose(0, 1)
            upper_summary = upper_summary + upper_outputs.sum(dim=1)
            upper_count = upper_count + pair_acoustic.shape[1]

        return (
            lower_hidden,
            upper_hidden,
            lower_summary + lower_outputs.sum(dim=1),
            upper_summary,
            count + total_steps,
            upper_count,
            pending_acoustic,
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (
            lower_hidden,
            upper_hidden,
            lower_summary,
            upper_summary,
            count,
            upper_count,
            pending_acoustic,
        ) = state
        del pending_acoustic
        features = torch.cat(
            (
                lower_summary / count.clamp_min(1.0),
                upper_summary / upper_count.clamp_min(1.0),
                lower_hidden[:, 0, :],
                upper_hidden[:, 0, :],
            ),
            dim=-1,
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(4, available_frames))


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
