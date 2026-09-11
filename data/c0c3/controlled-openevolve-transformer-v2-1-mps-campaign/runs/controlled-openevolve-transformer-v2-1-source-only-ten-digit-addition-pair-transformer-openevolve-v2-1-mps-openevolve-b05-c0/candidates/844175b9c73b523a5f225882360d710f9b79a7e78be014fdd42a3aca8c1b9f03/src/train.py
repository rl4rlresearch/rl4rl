"""Training entrypoint for smallest addition transformer.

Usage example:
  python -m src.train --run-name repro --train-steps 5000
"""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.data import (
    MAX_OPERAND,
    INPUT_LEN,
    VOCAB_SIZE,
    build_holdout_splits,
    encode_batch,
    pair_hash,
)
from src.eval import evaluate_exact_match
from src.model import ModelConfig, TinyDecoderLM, count_parameters


@dataclass
class TrainConfig:
    seed: int
    train_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    min_lr_ratio: float
    grad_clip: float
    eval_interval: int
    val_size: int
    test_size: int
    eval_batch_size: int
    run_name: str
    run_dir: str
    split_dir: str
    best_ckpt_out: str
    last_ckpt_out: str
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class TrainBatchSampler:
    def __init__(self, batch_size: int, seed: int, reserved_hashes: set):
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.reserved_hashes = reserved_hashes

    def sample_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)
        b = torch.randint(0, MAX_OPERAND, (self.batch_size,), generator=self.g, dtype=torch.int64)

        # Strictly avoid holdout leakage.
        for i in range(self.batch_size):
            ai = int(a[i].item())
            bi = int(b[i].item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
        return a, b

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.sample_pairs()
        return encode_batch(a, b)


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


class GaugeFixedAdamW:
    """AdamW on a bias quotient while retaining virtual full-bias moments."""

    def __init__(
        self,
        parameters,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.parameters = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            p: {
                "step": 0,
                "exp_avg": torch.zeros(p.numel() + 1, device=p.device, dtype=p.dtype),
                "exp_avg_sq": torch.zeros(p.numel() + 1, device=p.device, dtype=p.dtype),
            }
            for p in self.parameters
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for p in self.parameters:
            if p.grad is None:
                continue

            grad = p.grad.detach().reshape(-1)
            virtual_grad = torch.cat((grad, -grad.sum().reshape(1)))
            state = self.state[p]
            state["step"] += 1
            step = state["step"]

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(virtual_grad, alpha=1.0 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad, virtual_grad, value=1.0 - self.beta2
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
            direction = exp_avg / denom

            p.mul_(1.0 - self.lr * self.weight_decay)
            quotient_direction = direction[:-1] - direction[-1]
            p.add_(
                quotient_direction.view_as(p),
                alpha=-self.lr / bias_correction1,
            )


class TokenPositionGaugeAdamW:
    """AdamW with virtual coordinates for three embedding gauges."""

    def __init__(
        self,
        token_parameter,
        position_parameter,
        num_embeddings: int,
        embedding_dim: int,
        transfer_feature: int,
        position_embeddings: int,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.token_parameter = token_parameter
        self.position_parameter = position_parameter
        self.parameters = [token_parameter, position_parameter]
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.transfer_feature = transfer_feature
        self.position_embeddings = position_embeddings
        self.transfer_index = (
            (num_embeddings - 1) * embedding_dim + transfer_feature
        )
        self.global_index = num_embeddings * embedding_dim - 1
        self.fixed_indices = (self.transfer_index, self.global_index)
        self.position_global_index = (
            position_embeddings * embedding_dim - 1
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps

        full_token_numel = token_parameter.numel() + 2
        self.state = {
            "step": 0,
            "token_exp_avg": torch.zeros(
                full_token_numel,
                device=token_parameter.device,
                dtype=token_parameter.dtype,
            ),
            "token_exp_avg_sq": torch.zeros(
                full_token_numel,
                device=token_parameter.device,
                dtype=token_parameter.dtype,
            ),
            "position_exp_avg": torch.zeros(
                position_parameter.numel() + 1,
                device=position_parameter.device,
                dtype=position_parameter.dtype,
            ),
            "position_exp_avg_sq": torch.zeros(
                position_parameter.numel() + 1,
                device=position_parameter.device,
                dtype=position_parameter.dtype,
            ),
        }

    def _keep_mask(self) -> torch.Tensor:
        keep = torch.ones(
            self.token_parameter.numel() + 2,
            dtype=torch.bool,
            device=self.token_parameter.device,
        )
        keep[list(self.fixed_indices)] = False
        return keep

    def _position_keep_mask(self) -> torch.Tensor:
        keep = torch.ones(
            self.position_parameter.numel() + 1,
            dtype=torch.bool,
            device=self.position_parameter.device,
        )
        keep[self.position_global_index] = False
        return keep

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        if (
            self.token_parameter.grad is None
            or self.position_parameter.grad is None
        ):
            return

        keep = self._keep_mask()
        virtual_token_grad = self.token_parameter.grad.new_zeros(keep.numel())
        virtual_token_grad[keep] = (
            self.token_parameter.grad.detach().reshape(-1)
        )

        position_keep = self._position_keep_mask()
        virtual_position_grad = self.position_parameter.grad.new_zeros(
            position_keep.numel()
        )
        virtual_position_grad[position_keep] = (
            self.position_parameter.grad.detach().reshape(-1)
        )
        virtual_position_grad[self.position_global_index] = (
            -virtual_position_grad.sum()
        )

        token_matrix = virtual_token_grad.view(
            self.num_embeddings,
            self.embedding_dim,
        )
        position_matrix = virtual_position_grad.view(
            self.position_embeddings,
            self.embedding_dim,
        )
        transfer_grad = (
            position_matrix[:, self.transfer_feature].sum()
            - token_matrix[:, self.transfer_feature].sum()
        )
        virtual_token_grad[self.transfer_index] = transfer_grad
        virtual_token_grad[self.global_index] = -virtual_token_grad.sum()

        state = self.state
        state["step"] += 1
        step = state["step"]

        token_exp_avg = state["token_exp_avg"]
        token_exp_avg_sq = state["token_exp_avg_sq"]
        token_exp_avg.mul_(self.beta1).add_(
            virtual_token_grad,
            alpha=1.0 - self.beta1,
        )
        token_exp_avg_sq.mul_(self.beta2).addcmul_(
            virtual_token_grad,
            virtual_token_grad,
            value=1.0 - self.beta2,
        )

        position_exp_avg = state["position_exp_avg"]
        position_exp_avg_sq = state["position_exp_avg_sq"]
        position_exp_avg.mul_(self.beta1).add_(
            virtual_position_grad,
            alpha=1.0 - self.beta1,
        )
        position_exp_avg_sq.mul_(self.beta2).addcmul_(
            virtual_position_grad,
            virtual_position_grad,
            value=1.0 - self.beta2,
        )

        bias_correction1 = 1.0 - self.beta1**step
        bias_correction2 = 1.0 - self.beta2**step
        token_direction = token_exp_avg / (
            token_exp_avg_sq.sqrt().div(
                math.sqrt(bias_correction2)
            ).add(self.eps)
        )
        position_direction = position_exp_avg / (
            position_exp_avg_sq.sqrt().div(
                math.sqrt(bias_correction2)
            ).add(self.eps)
        )

        global_direction = token_direction[self.global_index]
        transfer_direction = token_direction[self.transfer_index]
        quotient_token = token_direction.view(
            self.num_embeddings,
            self.embedding_dim,
        ) - global_direction
        quotient_token[:, self.transfer_feature] = (
            token_direction.view(
                self.num_embeddings,
                self.embedding_dim,
            )[:, self.transfer_feature]
            - transfer_direction
        )
        quotient_token = quotient_token.reshape(-1)[keep]

        position_global_direction = position_direction[
            self.position_global_index
        ]
        quotient_position_full = position_direction.view(
            self.position_embeddings,
            self.embedding_dim,
        ) - position_global_direction
        quotient_position_full[:, self.transfer_feature].add_(
            transfer_direction - global_direction
        )
        quotient_position = quotient_position_full.reshape(-1)[position_keep]

        self.token_parameter.mul_(1.0 - self.lr * self.weight_decay)
        self.position_parameter.mul_(1.0 - self.lr * self.weight_decay)
        self.token_parameter.add_(
            quotient_token.view_as(self.token_parameter),
            alpha=-self.lr / bias_correction1,
        )
        self.position_parameter.add_(
            quotient_position.view_as(self.position_parameter),
            alpha=-self.lr / bias_correction1,
        )


class KeyGaugeAdamW:
    """AdamW with a virtual coordinate for a LayerNorm-null key direction."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [parameter for parameter, _, _, _ in self.gauges]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    parameter.numel() + len(fixed_rows),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + len(fixed_rows),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter, _, _, fixed_rows in self.gauges
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for parameter, ln_scale, d_model, fixed_rows in self.gauges:
            if parameter.grad is None:
                continue

            fixed_indices = tuple(
                row * d_model + d_model - 1 for row in fixed_rows
            )
            full_numel = parameter.numel() + len(fixed_indices)
            keep = torch.ones(
                full_numel,
                dtype=torch.bool,
                device=parameter.device,
            )
            keep[list(fixed_indices)] = False

            grad = parameter.grad.detach().reshape(-1)
            virtual_grad = grad.new_zeros(full_numel)
            virtual_grad[keep] = grad

            scale = ln_scale.detach().reshape(-1)
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                virtual_grad[fixed_index] = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()

            state = self.state[parameter]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                virtual_grad,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad,
                virtual_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(self.eps)
            direction = exp_avg / denom

            quotient_full = direction.clone()
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                quotient_full[row_start:fixed_index].sub_(
                    direction[fixed_index] * scale[-1] / scale[:-1]
                )
            quotient_direction = quotient_full[keep]

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                quotient_direction.view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )


class MLPOutputWeightGaugeAdamW:
    """AdamW with virtual coordinates for common-output weight directions."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [parameter for parameter, _, _, _, _ in self.gauges]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    parameter.numel() + len(fixed_rows),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + len(fixed_rows),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter, _, _, fixed_rows, _ in self.gauges
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for (
            parameter,
            out_features,
            in_features,
            fixed_rows,
            fixed_columns,
        ) in self.gauges:
            if parameter.grad is None:
                continue

            fixed_indices = tuple(
                row * in_features + column
                for row, column in zip(fixed_rows, fixed_columns)
            )
            full_numel = parameter.numel() + len(fixed_indices)
            keep = torch.ones(
                full_numel,
                dtype=torch.bool,
                device=parameter.device,
            )
            keep[list(fixed_indices)] = False

            virtual_grad = parameter.grad.new_zeros(full_numel)
            virtual_grad[keep] = parameter.grad.detach().reshape(-1)
            virtual_matrix = virtual_grad.view(out_features, in_features)
            for fixed_index, fixed_column in zip(
                fixed_indices,
                fixed_columns,
            ):
                virtual_grad[fixed_index] = -virtual_matrix[
                    :, fixed_column
                ].sum()

            state = self.state[parameter]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                virtual_grad,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad,
                virtual_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            direction = exp_avg / (
                exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )

            quotient_full = direction.clone()
            quotient_matrix = quotient_full.view(
                out_features,
                in_features,
            )
            for fixed_index, fixed_column in zip(
                fixed_indices,
                fixed_columns,
            ):
                quotient_matrix[:, fixed_column].sub_(
                    direction[fixed_index]
                )

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                quotient_full[keep].view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )


class ValueBiasGaugeAdamW:
    """Virtual AdamW for a value bias absorbed into the projection bias."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            bias: {
                "step": 0,
                "exp_avg": torch.zeros(
                    (),
                    device=bias.device,
                    dtype=bias.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    (),
                    device=bias.device,
                    dtype=bias.dtype,
                ),
            }
            for _, bias, _ in self.gauges
        }
        self.pending_offsets = []

    @torch.no_grad()
    def step(self) -> None:
        self.pending_offsets = []
        for projection, bias, feature_index in self.gauges:
            if bias.grad is None:
                continue

            reduced_grad = bias.grad.detach().reshape(-1)
            full_grad = torch.cat(
                (
                    reduced_grad,
                    -reduced_grad.sum().reshape(1),
                )
            )
            full_weight = projection.full_weight()
            virtual_grad = torch.dot(
                full_weight[:, feature_index].detach(),
                full_grad,
            )

            state = self.state[bias]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                virtual_grad,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad,
                virtual_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            direction = exp_avg / (
                exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )
            offset = -self.lr * direction / bias_correction1
            self.pending_offsets.append(
                (projection, bias, feature_index, offset)
            )

    @torch.no_grad()
    def project_biases(self) -> None:
        for projection, bias, feature_index, offset in self.pending_offsets:
            column = projection.full_weight()[:, feature_index]
            bias.add_((column[:-1] - column[-1]) * offset)
        self.pending_offsets = []


class AffineBiasGaugeAdamW:
    """AdamW with virtual LayerNorm biases absorbed into a downstream bias."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [parameter for parameter, _, _, _, _ in self.gauges]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    parameter.numel() + len(feature_indices),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + len(feature_indices),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter, _, _, feature_indices, _ in self.gauges
        }
        self.pending_offsets = []

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.pending_offsets = []
        for (
            parameter,
            downstream_weight,
            downstream_bias,
            feature_indices,
            omitted_positions,
        ) in self.gauges:
            if parameter.grad is None or downstream_bias.grad is None:
                continue

            grad = parameter.grad.detach().reshape(-1)
            virtual_grad = grad.new_zeros(
                grad.numel() + len(omitted_positions)
            )
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=virtual_grad.device,
            )
            keep[list(omitted_positions)] = False
            virtual_grad[keep] = grad
            for feature_index, omitted_position in zip(
                feature_indices,
                omitted_positions,
            ):
                virtual_grad[omitted_position] = torch.dot(
                    downstream_weight[:, feature_index].detach(),
                    downstream_bias.grad.detach(),
                )

            state = self.state[parameter]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                virtual_grad,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad,
                virtual_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            direction = exp_avg / (
                exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                direction[keep].view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )
            for feature_index, omitted_position in zip(
                feature_indices,
                omitted_positions,
            ):
                omitted_update = (
                    -self.lr
                    * direction[omitted_position]
                    / bias_correction1
                )
                self.pending_offsets.append(
                    (
                        downstream_weight,
                        downstream_bias,
                        feature_index,
                        omitted_update,
                    )
                )

    @torch.no_grad()
    def project_biases(self) -> None:
        for weight, bias, feature_index, offset in self.pending_offsets:
            bias.add_(weight[:, feature_index] * offset)
        self.pending_offsets = []


def clip_grad_norm_with_virtual_gauge(
    model: TinyDecoderLM,
    gauge_parameters,
    token_position_gauge,
    key_gauges,
    affine_bias_gauges,
    output_weight_gauges,
    value_bias_gauges,
    max_norm: float,
) -> None:
    active = [p for p in model.parameters() if p.grad is not None]
    if not active:
        return

    total_sq = torch.zeros(
        (),
        device=active[0].grad.device,
        dtype=torch.float32,
    )
    for p in active:
        total_sq.add_(p.grad.detach().float().pow(2).sum())

    # The omitted full-bias gradient is fixed by common-shift invariance.
    for p in gauge_parameters:
        if p.grad is not None:
            total_sq.add_(p.grad.detach().float().sum().pow(2))

    # Recover the omitted position and token gradients from their invariances.
    (
        token_parameter,
        position_parameter,
        num_embeddings,
        embedding_dim,
        transfer_feature,
        position_embeddings,
    ) = token_position_gauge
    if (
        token_parameter.grad is not None
        and position_parameter.grad is not None
    ):
        position_global_index = (
            position_embeddings * embedding_dim - 1
        )
        position_keep = torch.ones(
            position_parameter.numel() + 1,
            dtype=torch.bool,
            device=position_parameter.grad.device,
        )
        position_keep[position_global_index] = False
        virtual_position_grad = (
            position_parameter.grad.detach().float().new_zeros(
                position_parameter.numel() + 1
            )
        )
        virtual_position_grad[position_keep] = (
            position_parameter.grad.detach().reshape(-1).float()
        )
        virtual_position_grad[position_global_index] = (
            -virtual_position_grad.sum()
        )

        transfer_index = (
            (num_embeddings - 1) * embedding_dim + transfer_feature
        )
        global_index = num_embeddings * embedding_dim - 1
        fixed_indices = (transfer_index, global_index)
        virtual_token_grad = token_parameter.grad.detach().float().new_zeros(
            token_parameter.numel() + 2
        )
        keep = torch.ones(
            virtual_token_grad.numel(),
            dtype=torch.bool,
            device=virtual_token_grad.device,
        )
        keep[list(fixed_indices)] = False
        virtual_token_grad[keep] = (
            token_parameter.grad.detach().reshape(-1).float()
        )
        token_matrix = virtual_token_grad.view(
            num_embeddings,
            embedding_dim,
        )
        position_matrix = virtual_position_grad.view(
            position_embeddings,
            embedding_dim,
        )
        virtual_token_grad[transfer_index] = (
            position_matrix[:, transfer_feature].sum()
            - token_matrix[:, transfer_feature].sum()
        )
        virtual_token_grad[global_index] = -virtual_token_grad.sum()
        total_sq.add_(
            virtual_token_grad[list(fixed_indices)].pow(2).sum()
        )
        total_sq.add_(
            virtual_position_grad[position_global_index].pow(2)
        )

    # Recover the omitted key gradients from the LayerNorm-null directions.
    for parameter, ln_scale, d_model, fixed_rows in key_gauges:
        if parameter.grad is not None:
            fixed_indices = tuple(
                row * d_model + d_model - 1 for row in fixed_rows
            )
            grad = parameter.grad.detach().reshape(-1).float()
            virtual_grad = grad.new_zeros(grad.numel() + len(fixed_indices))
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=grad.device,
            )
            keep[list(fixed_indices)] = False
            virtual_grad[keep] = grad

            scale = ln_scale.detach().reshape(-1).float()
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                omitted = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()
                total_sq.add_(omitted.pow(2))

    # Recover each omitted LayerNorm-bias gradient through the downstream affine.
    for (
        parameter,
        downstream_weight,
        downstream_bias,
        feature_indices,
        _,
    ) in affine_bias_gauges:
        if parameter.grad is not None and downstream_bias.grad is not None:
            for feature_index in feature_indices:
                omitted = torch.dot(
                    downstream_weight[:, feature_index].detach().float(),
                    downstream_bias.grad.detach().float(),
                )
                total_sq.add_(omitted.pow(2))

    # Recover omitted output-weight gradients from common-shift invariance.
    for (
        parameter,
        out_features,
        in_features,
        fixed_rows,
        fixed_columns,
    ) in output_weight_gauges:
        if parameter.grad is not None:
            fixed_indices = tuple(
                row * in_features + column
                for row, column in zip(fixed_rows, fixed_columns)
            )
            virtual_grad = parameter.grad.detach().float().new_zeros(
                parameter.numel() + len(fixed_indices)
            )
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=virtual_grad.device,
            )
            keep[list(fixed_indices)] = False
            virtual_grad[keep] = parameter.grad.detach().reshape(-1).float()
            virtual_matrix = virtual_grad.view(out_features, in_features)
            for fixed_column in fixed_columns:
                omitted = -virtual_matrix[:, fixed_column].sum()
                total_sq.add_(omitted.pow(2))

    # Recover the omitted value-bias gradient through the output projection.
    for projection, bias, feature_index in value_bias_gauges:
        if bias.grad is not None:
            reduced_grad = bias.grad.detach().reshape(-1).float()
            full_grad = torch.cat(
                (
                    reduced_grad,
                    -reduced_grad.sum().reshape(1),
                )
            )
            omitted = torch.dot(
                projection.full_weight()[:, feature_index].detach().float(),
                full_grad,
            )
            total_sq.add_(omitted.pow(2))

    coefficient = min(
        1.0,
        max_norm / (float(total_sq.sqrt().item()) + 1e-6),
    )
    if coefficient < 1.0:
        for p in active:
            p.grad.mul_(coefficient)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv_header(path: Path, header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv(path: Path, row: List) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Dict:
    device = torch.device(train_cfg.device)
    run_dir = Path(train_cfg.run_dir)
    split_dir = Path(train_cfg.split_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.csv"
    set_seed(train_cfg.seed)

    split_path = split_dir / f"holdout_v{train_cfg.val_size}_t{train_cfg.test_size}_seed{train_cfg.seed}.pt"
    splits = build_holdout_splits(train_cfg.val_size, train_cfg.test_size, train_cfg.seed, split_path)

    reserved_hashes = set()
    for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
    for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))

    val_a, val_b = splits["val_a"], splits["val_b"]

    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    token_position_gauge = (
        model.token_emb.weight,
        model.pos_emb.weight,
        model.token_emb.num_embeddings,
        model.token_emb.embedding_dim,
        model.token_emb.transfer_feature,
        model.pos_emb.num_embeddings,
    )
    gauge_parameters = [
        *[block.attn.proj_bias for block in model.blocks],
        *[block.mlp.fc2.bias for block in model.blocks],
    ]
    key_gauges = [
        (
            block.attn.qkv.weight,
            block.ln1.weight,
            model_cfg.d_model,
            block.attn.qkv.fixed_rows,
        )
        for block in model.blocks
    ]
    output_weight_gauges = [
        (
            block.attn.proj.weight,
            block.attn.proj.out_features,
            block.attn.proj.in_features,
            block.attn.proj.fixed_weight_rows,
            block.attn.proj.fixed_weight_columns,
        )
        for block in model.blocks
    ] + [
        (
            block.mlp.fc2.weight,
            block.mlp.fc2.out_features,
            block.mlp.fc2.in_features,
            (block.mlp.fc2.fixed_weight_row,),
            (block.mlp.fc2.fixed_weight_column,),
        )
        for block in model.blocks
    ]
    value_bias_gauges = [
        (
            block.attn.proj,
            block.attn.proj_bias,
            block.attn.virtual_v_bias_feature,
        )
        for block in model.blocks
    ]
    affine_bias_gauges = [
        (
            block.ln2.bias,
            block.mlp.fc1.weight,
            block.mlp.fc1.bias,
            (1, 6),
            (0, 2),
        )
        for block in model.blocks
    ]
    gauge_parameter_ids = {
        id(p) for p in [
            token_position_gauge[0],
            token_position_gauge[1],
            *gauge_parameters,
            *[parameter for parameter, _, _, _ in key_gauges],
            *[
                parameter
                for parameter, _, _, _, _ in output_weight_gauges
            ],
            *[
                parameter
                for parameter, _, _, _, _ in affine_bias_gauges
            ],
        ]
    }
    ordinary_parameters = [
        p for p in model.parameters() if id(p) not in gauge_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        ordinary_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizer = GaugeFixedAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    token_position_optimizer = TokenPositionGaugeAdamW(
        *token_position_gauge,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    key_optimizer = KeyGaugeAdamW(
        key_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    output_weight_optimizer = MLPOutputWeightGaugeAdamW(
        output_weight_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    affine_bias_optimizer = AffineBiasGaugeAdamW(
        affine_bias_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    value_bias_optimizer = ValueBiasGaugeAdamW(
        value_bias_gauges,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
    save_csv_header(metrics_path, ["step", "train_loss", "val_exact", "val_token_acc", "lr", "elapsed_sec"])

    best_val = -1.0
    best_step = -1
    t0 = time.time()

    print(f"Run: {train_cfg.run_name}")
    print(f"Params: {params}")
    print(f"Device: {device}")

    for step in range(train_cfg.train_steps):
        model.train()
        x, y = sampler.sample_batch()
        x = x.to(device)
        y = y.to(device)

        lr_now = cosine_lr(step, train_cfg.train_steps, train_cfg.lr, train_cfg.warmup_steps, train_cfg.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        gauge_optimizer.lr = lr_now
        token_position_optimizer.lr = lr_now
        key_optimizer.lr = lr_now
        output_weight_optimizer.lr = lr_now
        affine_bias_optimizer.lr = lr_now
        value_bias_optimizer.lr = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        token_position_optimizer.zero_grad(set_to_none=True)
        key_optimizer.zero_grad(set_to_none=True)
        output_weight_optimizer.zero_grad(set_to_none=True)
        affine_bias_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_virtual_gauge(
                model,
                gauge_parameters,
                token_position_gauge,
                key_gauges,
                affine_bias_gauges,
                output_weight_gauges,
                value_bias_gauges,
                train_cfg.grad_clip,
            )
        value_bias_optimizer.step()
        key_optimizer.step()
        output_weight_optimizer.step()
        affine_bias_optimizer.step()
        optimizer.step()
        affine_bias_optimizer.project_biases()
        gauge_optimizer.step()
        value_bias_optimizer.project_biases()
        token_position_optimizer.step()

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
            val_exact, val_tok = evaluate_exact_match(model, val_a, val_b, train_cfg.eval_batch_size, device)
            elapsed = time.time() - t0
            train_loss = float(loss.item())
            append_csv(metrics_path, [step, train_loss, val_exact, val_tok, lr_now, elapsed])
            print(
                f"step={step:6d} loss={train_loss:.4f} val_exact={val_exact:.5f} "
                f"val_tok={val_tok:.5f} lr={lr_now:.2e} t={elapsed:.1f}s"
            )

            if val_exact > best_val:
                best_val = val_exact
                best_step = step
                Path(train_cfg.best_ckpt_out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": asdict(model_cfg),
                        "train_config": asdict(train_cfg),
                        "step": step,
                        "val_exact": val_exact,
                        "params": params,
                    },
                    train_cfg.best_ckpt_out,
                )

    Path(train_cfg.last_ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "step": train_cfg.train_steps - 1,
            "val_exact": best_val,
            "params": params,
        },
        train_cfg.last_ckpt_out,
    )

    summary = {
        "run_name": train_cfg.run_name,
        "params": params,
        "best_val_exact": best_val,
        "best_step": best_step,
        "train_steps": train_cfg.train_steps,
        "elapsed_sec": time.time() - t0,
        "metrics_csv": str(metrics_path),
        "best_ckpt": str(train_cfg.best_ckpt_out),
        "last_ckpt": str(train_cfg.last_ckpt_out),
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "config.json", {"model": asdict(model_cfg), "train": asdict(train_cfg)})
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train smallest addition transformer")

    # run/output
    p.add_argument("--run-name", type=str, default="repro_l1_d8_ff12")
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/repro_l1_d8_ff12"))
    p.add_argument("--split-dir", type=Path, default=Path("results/data"))
    p.add_argument("--best-ckpt-out", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--last-ckpt-out", type=Path, default=Path("checkpoints/last.pt"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=123)

    # model
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.0)

    # optimization
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=300)

    # eval split
    p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--test-size", type=int, default=10000)
    p.add_argument("--eval-batch-size", type=int, default=512)

    args = p.parse_args()

    model_cfg = ModelConfig(
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=INPUT_LEN,
        vocab_size=VOCAB_SIZE,
    )

    train_cfg = TrainConfig(
        seed=args.seed,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        val_size=args.val_size,
        test_size=args.test_size,
        eval_batch_size=args.eval_batch_size,
        run_name=args.run_name,
        run_dir=str(args.run_dir),
        split_dir=str(args.split_dir),
        best_ckpt_out=str(args.best_ckpt_out),
        last_ckpt_out=str(args.last_ckpt_out),
        device=args.device,
    )

    summary = train(model_cfg, train_cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
