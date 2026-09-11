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
    """A causal multirate GRU with fine and pair-pooled temporal states."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.slow_gru = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(152, 8)
        self.endpoint_classifier = nn.Linear(152, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fast_hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        slow_hidden = torch.zeros(batch_size, 1, 40, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 40, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            fast_hidden,
            slow_hidden,
            fast_summary,
            slow_summary,
            pending,
            count,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            slow_summary,
            pending,
            count,
        ) = state
        normalized = self.input_norm(frame)
        fast_output, fast_hidden = self.fast_gru(
            normalized.unsqueeze(1),
            fast_hidden.transpose(0, 1).contiguous(),
        )
        fast_summary = fast_summary + fast_output[:, 0, :]

        if int(count[0, 0].item()) % 2 == 1:
            paired = 0.5 * (pending + normalized)
            slow_output, slow_hidden = self.slow_gru(
                paired.unsqueeze(1),
                slow_hidden.transpose(0, 1).contiguous(),
            )
            slow_summary = slow_summary + slow_output[:, 0, :]

        return (
            fast_hidden.transpose(0, 1),
            slow_hidden.transpose(0, 1),
            fast_summary,
            slow_summary,
            normalized,
            count + 1.0,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            slow_summary,
            pending,
            count,
        ) = state
        normalized = self.input_norm(frames)
        fast_outputs, fast_hidden = self.fast_gru(
            normalized,
            fast_hidden.transpose(0, 1).contiguous(),
        )
        fast_summary = fast_summary + fast_outputs.sum(dim=1)

        if int(count[0, 0].item()) % 2 == 0:
            pair_count = frames.shape[1] // 2
            paired = 0.5 * (
                normalized[:, : 2 * pair_count : 2, :]
                + normalized[:, 1 : 2 * pair_count : 2, :]
            )
        else:
            pair_parts = [
                0.5 * (pending.unsqueeze(1) + normalized[:, :1, :])
            ]
            remaining_pairs = (frames.shape[1] - 1) // 2
            if remaining_pairs > 0:
                pair_parts.append(
                    0.5
                    * (
                        normalized[:, 1 : 1 + 2 * remaining_pairs : 2, :]
                        + normalized[:, 2 : 1 + 2 * remaining_pairs : 2, :]
                    )
                )
            paired = torch.cat(pair_parts, dim=1)

        if paired.shape[1] > 0:
            slow_outputs, slow_hidden = self.slow_gru(
                paired,
                slow_hidden.transpose(0, 1).contiguous(),
            )
            slow_summary = slow_summary + slow_outputs.sum(dim=1)

        return (
            fast_hidden.transpose(0, 1),
            slow_hidden.transpose(0, 1),
            fast_summary,
            slow_summary,
            normalized[:, -1, :],
            count + frames.shape[1],
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
        ],
    ) -> torch.Tensor:
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            slow_summary,
            _pending,
            count,
        ) = state
        slow_count = torch.floor(0.5 * count).clamp_min(1.0)
        mean_output = torch.cat(
            (
                fast_summary / count.clamp_min(1.0),
                slow_summary / slow_count,
            ),
            dim=1,
        )
        endpoint = torch.cat(
            (fast_hidden[:, 0, :], slow_hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)

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
