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

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class KeywordStackedRNN(nn.Module):
    """A bottlenecked two-level tanh recurrence with multi-scale readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.lower_rnn = nn.RNN(
            20, 64, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.bridge_norm = nn.LayerNorm(64)
        self.upper_rnn = nn.RNN(
            64, 128, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.output_norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(320, 8)

        for recurrent in (self.lower_rnn, self.upper_rnn):
            nn.init.xavier_uniform_(recurrent.weight_ih_l0)
            nn.init.orthogonal_(recurrent.weight_hh_l0)
            nn.init.zeros_(recurrent.bias_ih_l0)
            nn.init.zeros_(recurrent.bias_hh_l0)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lower_hidden = torch.zeros(
            batch_size, 1, 64, device=device, dtype=dtype
        )
        upper_hidden = torch.zeros(
            batch_size, 1, 128, device=device, dtype=dtype
        )
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return lower_hidden, upper_hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lower_hidden, upper_hidden, summary, count = state
        lower_output, lower_hidden = self.lower_rnn(
            self.input_norm(frame).unsqueeze(1),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_output, upper_hidden = self.upper_rnn(
            self.bridge_norm(lower_output),
            upper_hidden.transpose(0, 1).contiguous(),
        )
        upper_output = upper_output[:, 0, :]
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            summary + upper_output,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lower_hidden, upper_hidden, summary, count = state
        lower_outputs, lower_hidden = self.lower_rnn(
            self.input_norm(frames),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_outputs, upper_hidden = self.upper_rnn(
            self.bridge_norm(lower_outputs),
            upper_hidden.transpose(0, 1).contiguous(),
        )
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            summary + upper_outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        lower_hidden, upper_hidden, summary, count = state
        features = torch.cat(
            (
                self.output_norm(summary / count.clamp_min(1.0)),
                self.output_norm(upper_hidden[:, 0, :]),
                self.bridge_norm(lower_hidden[:, 0, :]),
            ),
            dim=-1,
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))


def build_model() -> nn.Module:
    return KeywordStackedRNN()


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
