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


class KeywordRNN(nn.Module):
    """Three wider, orthogonally stabilized causal tanh RNN streams."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.rnn_heads = nn.ModuleList(
            [
                nn.RNN(
                    20,
                    48,
                    num_layers=1,
                    nonlinearity="tanh",
                    batch_first=True,
                )
                for _ in range(3)
            ]
        )
        self.output_norms = nn.ModuleList([nn.LayerNorm(48) for _ in range(3)])
        for rnn in self.rnn_heads:
            nn.init.xavier_uniform_(rnn.weight_ih_l0)
            nn.init.orthogonal_(rnn.weight_hh_l0)
            nn.init.zeros_(rnn.bias_ih_l0)
            nn.init.zeros_(rnn.bias_hh_l0)
        self.classifier = nn.Linear(432, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_0 = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 144, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 144, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_0, hidden_1, hidden_2, summary, peak, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_0, hidden_1, hidden_2, summary, peak, count = state
        hidden_states = (hidden_0, hidden_1, hidden_2)
        normalized = self.input_norm(frame).unsqueeze(1)
        outputs = []
        next_hidden = []
        for rnn, output_norm, hidden in zip(
            self.rnn_heads, self.output_norms, hidden_states
        ):
            head_output, head_hidden = rnn(
                normalized,
                hidden.unsqueeze(0).contiguous(),
            )
            outputs.append(output_norm(head_output[:, 0, :]))
            next_hidden.append(head_hidden[0])
        output = torch.cat(outputs, dim=-1)
        return (
            *next_hidden,
            summary + output,
            torch.maximum(peak, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_0, hidden_1, hidden_2, summary, peak, count = state
        hidden_states = (hidden_0, hidden_1, hidden_2)
        normalized = self.input_norm(frames)
        head_outputs = []
        next_hidden = []
        for rnn, output_norm, hidden in zip(
            self.rnn_heads, self.output_norms, hidden_states
        ):
            output, head_hidden = rnn(
                normalized,
                hidden.unsqueeze(0).contiguous(),
            )
            head_outputs.append(output_norm(output))
            next_hidden.append(head_hidden[0])
        outputs = torch.cat(head_outputs, dim=-1)
        return (
            *next_hidden,
            summary + outputs.sum(dim=1),
            torch.maximum(peak, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        hidden_0, hidden_1, hidden_2, summary, peak, count = state
        hidden_states = (hidden_0, hidden_1, hidden_2)
        endpoint = torch.cat(
            [
                output_norm(hidden)
                for output_norm, hidden in zip(
                    self.output_norms, hidden_states
                )
            ],
            dim=-1,
        )
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((endpoint, mean_output, peak), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 3))


def build_model() -> nn.Module:
    return KeywordRNN()


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
