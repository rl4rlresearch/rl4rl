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
    """A causal hierarchy of local acoustic and command-level GRUs."""

    def __init__(self) -> None:
        super().__init__()
        self.chunk_size = 4
        self.input_norm = nn.LayerNorm(20)
        self.fast_gru = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.slow_input_norm = nn.LayerNorm(48)
        self.slow_gru = nn.GRU(48, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(208, 8)

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
        fast_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        slow_hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        frame_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        slow_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            fast_hidden,
            slow_hidden,
            fast_summary,
            slow_summary,
            frame_count,
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
            frame_count,
            slow_count,
        ) = state
        fast_output, fast_hidden_t = self.fast_gru(
            self.input_norm(frame).unsqueeze(1),
            fast_hidden.transpose(0, 1).contiguous(),
        )
        fast_output = fast_output[:, 0, :]
        fast_hidden = fast_hidden_t.transpose(0, 1)
        fast_summary = fast_summary + fast_output
        frame_count = frame_count + 1.0

        if int(frame_count[0, 0].item()) % self.chunk_size == 0:
            slow_output, slow_hidden_t = self.slow_gru(
                self.slow_input_norm(fast_output).unsqueeze(1),
                slow_hidden.transpose(0, 1).contiguous(),
            )
            slow_output = slow_output[:, 0, :]
            slow_hidden = slow_hidden_t.transpose(0, 1)
            slow_summary = slow_summary + slow_output
            slow_count = slow_count + 1.0
            fast_hidden = torch.zeros_like(fast_hidden)

        return (
            fast_hidden,
            slow_hidden,
            fast_summary,
            slow_summary,
            frame_count,
            slow_count,
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
            frame_count,
            slow_count,
        ) = state
        length = frames.shape[1]
        fresh_aligned_state = (
            length % self.chunk_size == 0
            and bool(torch.all(frame_count == 0).item())
            and bool(torch.all(fast_hidden == 0).item())
        )
        if not fresh_aligned_state:
            current = state
            for index in range(length):
                current = self.recurrent_step(frames[:, index, :], current)
            return current

        batch_size = frames.shape[0]
        chunks = length // self.chunk_size
        normalized = self.input_norm(frames)
        chunked = normalized.reshape(
            batch_size * chunks, self.chunk_size, 20
        )
        fast_initial = frames.new_zeros(1, batch_size * chunks, 48)
        local_outputs, _ = self.fast_gru(chunked, fast_initial)
        local_outputs = local_outputs.reshape(
            batch_size, chunks, self.chunk_size, 48
        )
        chunk_endpoints = local_outputs[:, :, -1, :]

        slow_outputs, slow_hidden_t = self.slow_gru(
            self.slow_input_norm(chunk_endpoints),
            slow_hidden.transpose(0, 1).contiguous(),
        )
        return (
            torch.zeros_like(fast_hidden),
            slow_hidden_t.transpose(0, 1),
            fast_summary + local_outputs.sum(dim=(1, 2)),
            slow_summary + slow_outputs.sum(dim=1),
            frame_count + length,
            slow_count + chunks,
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
            frame_count,
            slow_count,
        ) = state
        del fast_hidden
        fast_mean = fast_summary / frame_count.clamp_min(1.0)
        slow_mean = slow_summary / slow_count.clamp_min(1.0)
        features = torch.cat(
            (fast_mean, slow_hidden[:, 0, :], slow_mean), dim=-1
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))


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
