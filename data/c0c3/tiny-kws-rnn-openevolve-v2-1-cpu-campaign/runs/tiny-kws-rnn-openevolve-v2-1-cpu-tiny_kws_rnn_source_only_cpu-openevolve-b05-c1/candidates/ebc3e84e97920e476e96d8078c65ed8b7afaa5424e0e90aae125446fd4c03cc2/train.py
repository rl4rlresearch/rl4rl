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
    """A two-timescale causal recurrent keyword model."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.local_rnn = nn.RNN(
            20, 64, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.block_norm = nn.LayerNorm(64)
        self.global_gru = nn.GRU(64, 64, num_layers=1, batch_first=True)
        self.readout_norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(128, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        local_hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        global_hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return local_hidden, global_hidden, summary, block_sum, count

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
        local_hidden, global_hidden, summary, block_sum, count = state
        local_output, local_hidden_t = self.local_rnn(
            self.input_norm(frame).unsqueeze(1),
            local_hidden.transpose(0, 1).contiguous(),
        )
        local_output = local_output[:, 0, :]
        local_hidden = local_hidden_t.transpose(0, 1)
        block_sum = block_sum + local_output

        if int(count[0, 0].item()) % 4 == 3:
            block_descriptor = local_output + 0.25 * block_sum
            _, global_hidden_t = self.global_gru(
                self.block_norm(block_descriptor).unsqueeze(1),
                global_hidden.transpose(0, 1).contiguous(),
            )
            global_hidden = global_hidden_t.transpose(0, 1)
            local_hidden = torch.zeros_like(local_hidden)
            block_sum = torch.zeros_like(block_sum)

        return (
            local_hidden,
            global_hidden,
            summary + local_output,
            block_sum,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        local_hidden, global_hidden, summary, block_sum, count = state
        time_steps = frames.shape[1]
        phase = int(count[0, 0].item()) % 4

        if phase != 0 or time_steps % 4 != 0:
            current_state = state
            for index in range(time_steps):
                current_state = self.recurrent_step(
                    frames[:, index, :], current_state
                )
            return current_state

        batch_size = frames.shape[0]
        blocks = time_steps // 4
        grouped_frames = self.input_norm(frames).reshape(
            batch_size * blocks, 4, 20
        )
        local_outputs, _ = self.local_rnn(grouped_frames)
        local_outputs = local_outputs.reshape(batch_size, blocks, 4, 64)

        block_descriptors = (
            local_outputs[:, :, -1, :] + local_outputs.mean(dim=2)
        )
        _, global_hidden_t = self.global_gru(
            self.block_norm(block_descriptors),
            global_hidden.transpose(0, 1).contiguous(),
        )

        return (
            torch.zeros_like(local_hidden),
            global_hidden_t.transpose(0, 1),
            summary + local_outputs.sum(dim=(1, 2)),
            torch.zeros_like(block_sum),
            count + time_steps,
        )

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
        _, global_hidden, summary, _, count = state
        mean_local = summary / count.clamp_min(1.0)
        readout = torch.cat((mean_local, global_hidden[:, 0, :]), dim=-1)
        return self.classifier(self.readout_norm(readout))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(32, available_frames)
        return [
            i * (available_frames - 1) // (steps - 1)
            for i in range(steps)
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
