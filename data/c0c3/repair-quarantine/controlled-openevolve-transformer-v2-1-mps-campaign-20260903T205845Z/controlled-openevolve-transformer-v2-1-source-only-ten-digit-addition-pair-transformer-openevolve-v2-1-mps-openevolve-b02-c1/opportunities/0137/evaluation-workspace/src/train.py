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


class QuotientAdamW:
    """AdamW on stored gauge differences with one virtual reference coordinate."""

    def __init__(self, model: TinyDecoderLM, lr: float, weight_decay: float):
        self.all_params = list(model.parameters())
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (head_bias, 0)
            for block in model.blocks
            for head_bias in block.attn.rel_bias
        ]
        self.score_specs = [
            (
                block.attn.qkv.query_weight,
                block.attn.qkv.key_tail,
                block.attn.qkv.query_bias,
                block.attn.qkv,
            )
            for block in model.blocks
        ]
        self.attention_specs = [
            (
                block.attn.qkv.value_tail,
                block.attn.qkv,
                block.attn.proj.weight,
                block.attn.proj.bias,
            )
            for block in model.blocks
        ]
        self.gauge_params = [param for param, _ in self.gauge_specs]
        self.score_params = [
            param
            for query_weight, key_tail, query_bias, _
            in self.score_specs
            for param in (query_weight, key_tail, query_bias)
        ]
        self.attention_params = [
            param
            for weight, _, proj_weight, folded_bias
            in self.attention_specs
            for param in (weight, proj_weight, folded_bias)
        ]
        excluded_ids = {
            id(param)
            for param in (
                self.gauge_params
                + self.score_params
                + self.attention_params
            )
        }
        self.custom_param_ids = {
            id(param)
            for param in self.score_params + self.attention_params
        }
        ordinary_params = [
            param
            for param in self.all_params
            if id(param) not in excluded_ids
        ]

        self.base = torch.optim.AdamW(
            ordinary_params, lr=lr, weight_decay=weight_decay
        )
        self.param_groups = self.base.param_groups
        self.gauge_states = []
        for param, axis in self.gauge_specs:
            full_shape = list(param.shape)
            full_shape[axis] += 1
            self.gauge_states.append(
                {
                    "step": 0,
                    "exp_avg": param.new_zeros(full_shape),
                    "exp_avg_sq": param.new_zeros(full_shape),
                }
            )

        self.score_states = []
        for query_param, _, _, qkv in self.score_specs:
            query_weight = qkv._initial_query_weight.to(
                device=query_param.device, dtype=query_param.dtype
            )
            key_weight = qkv._initial_key_weight.to(
                device=query_param.device, dtype=query_param.dtype
            )
            query_bias = qkv._initial_query_bias.to(
                device=query_param.device, dtype=query_param.dtype
            )
            delattr(qkv, "_initial_query_weight")
            delattr(qkv, "_initial_key_weight")
            delattr(qkv, "_initial_query_bias")

            full_shape = list(query_weight.shape)
            full_shape[1] += 1
            self.score_states.append(
                {
                    "step": 0,
                    "query_weight": query_weight,
                    "key_weight": key_weight,
                    "query_bias": query_bias,
                    "exp_avg_query": query_weight.new_zeros(full_shape),
                    "exp_avg_sq_query": query_weight.new_zeros(full_shape),
                    "exp_avg_key": key_weight.new_zeros(full_shape),
                    "exp_avg_sq_key": key_weight.new_zeros(full_shape),
                    "exp_avg_bias": query_bias.new_zeros(query_bias.shape),
                    "exp_avg_sq_bias": query_bias.new_zeros(
                        query_bias.shape
                    ),
                }
            )

        self.attention_states = []
        for weight, qkv, proj_weight, _ in self.attention_specs:
            full_weight = qkv._initial_value_weight.to(
                device=weight.device, dtype=weight.dtype
            )
            virtual_proj_weight = qkv._initial_proj_weight.to(
                device=proj_weight.device, dtype=proj_weight.dtype
            )
            delattr(qkv, "_initial_value_weight")
            delattr(qkv, "_initial_proj_weight")
            full_shape = full_weight.shape
            width = full_shape[1]
            full_proj_shape = list(virtual_proj_weight.shape)
            full_proj_shape[0] += 1
            self.attention_states.append(
                {
                    "step": 0,
                    "full_weight": full_weight,
                    "proj_weight": virtual_proj_weight,
                    "scale": weight.new_ones(width),
                    "shift": weight.new_zeros(width),
                    "full_bias": weight.new_zeros(width),
                    "exp_avg_weight": weight.new_zeros(full_shape),
                    "exp_avg_sq_weight": weight.new_zeros(full_shape),
                    "exp_avg_scale": weight.new_zeros(width),
                    "exp_avg_sq_scale": weight.new_zeros(width),
                    "exp_avg_shift": weight.new_zeros(width),
                    "exp_avg_sq_shift": weight.new_zeros(width),
                    "exp_avg_bias": weight.new_zeros(width),
                    "exp_avg_sq_bias": weight.new_zeros(width),
                    "exp_avg_proj": proj_weight.new_zeros(
                        full_proj_shape
                    ),
                    "exp_avg_sq_proj": proj_weight.new_zeros(
                        full_proj_shape
                    ),
                }
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)
        for param in (
            self.gauge_params + self.score_params + self.attention_params
        ):
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    @staticmethod
    def _full_score_grads(
        query_param, key_tail_param, bias_param, state
    ):
        canonical_query_grad = (
            torch.zeros_like(query_param)
            if query_param.grad is None
            else query_param.grad.detach()
        )
        stored_key_tail_grad = (
            torch.zeros_like(key_tail_param)
            if key_tail_param.grad is None
            else key_tail_param.grad.detach()
        )
        canonical_bias_grad = (
            torch.zeros_like(bias_param)
            if bias_param.grad is None
            else bias_param.grad.detach()
        )

        n_head, head_dim, reduced_width = query_param.shape
        tail_width = reduced_width - head_dim
        learned_boundary = (
            (n_head * head_dim - 2) * tail_width
        )
        canonical_key_tail_grad = torch.cat(
            [
                stored_key_tail_grad[:learned_boundary],
                stored_key_tail_grad.new_zeros(3),
                stored_key_tail_grad[learned_boundary:],
                stored_key_tail_grad.new_zeros(tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
        detached_key_tail = key_tail_param.detach()
        canonical_key_tail = torch.cat(
            [
                detached_key_tail[:learned_boundary],
                detached_key_tail.new_zeros(3),
                detached_key_tail[learned_boundary:],
                detached_key_tail.new_zeros(tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
        virtual_key_heads = state["key_weight"].view(
            n_head, head_dim, reduced_width
        )
        key_basis = virtual_key_heads[..., :head_dim]

        # Recover the gradient of the fixed identity key block from the
        # exact GL(head_dim) factorization invariance.
        canonical_key_prefix_grad = torch.matmul(
            query_param, canonical_query_grad.transpose(-1, -2)
        )
        canonical_key_prefix_grad = canonical_key_prefix_grad + (
            bias_param.unsqueeze(-1)
            * canonical_bias_grad.unsqueeze(-2)
        )
        canonical_key_prefix_grad = canonical_key_prefix_grad - (
            torch.matmul(
                canonical_key_tail_grad,
                canonical_key_tail.transpose(-1, -2),
            )
        )
        canonical_key_grad = torch.cat(
            [canonical_key_prefix_grad, canonical_key_tail_grad],
            dim=-1,
        )

        # Map canonical-factor gradients back to the virtual Q/K factors
        # whose full-width AdamW moments are retained.
        query_grad = torch.matmul(
            key_basis, canonical_query_grad
        )
        key_grad = torch.linalg.solve(
            key_basis.transpose(-1, -2), canonical_key_grad
        )
        bias_grad = torch.matmul(
            key_basis, canonical_bias_grad.unsqueeze(-1)
        ).squeeze(-1)

        query_grad = query_grad.reshape_as(state["query_weight"])
        key_grad = key_grad.reshape_as(state["key_weight"])
        bias_grad = bias_grad.reshape_as(state["query_bias"])
        full_query_grad = torch.cat(
            [query_grad, -query_grad.sum(dim=1, keepdim=True)], dim=1
        )
        full_key_grad = torch.cat(
            [key_grad, -key_grad.sum(dim=1, keepdim=True)], dim=1
        )
        return full_query_grad, full_key_grad, bias_grad

    @staticmethod
    def _full_attention_grads(
        weight_param, proj_weight_param, folded_bias_param, state
    ):
        canonical_tail_grad = (
            torch.zeros_like(weight_param)
            if weight_param.grad is None
            else weight_param.grad.detach()
        )
        canonical_proj_grad = (
            torch.zeros_like(proj_weight_param)
            if proj_weight_param.grad is None
            else proj_weight_param.grad.detach()
        )
        folded_grad = (
            torch.zeros_like(folded_bias_param)
            if folded_bias_param.grad is None
            else folded_bias_param.grad.detach()
        )

        n_head, head_dim, tail_width = weight_param.shape
        reduced_width = head_dim + tail_width
        scaled_weight = (
            state["full_weight"] * state["scale"].unsqueeze(0)
        )
        effective_weight = (
            scaled_weight[:, :-1] - scaled_weight[:, -1:]
        )
        value_heads = effective_weight.view(
            n_head, head_dim, reduced_width
        )
        value_basis = value_heads[..., :head_dim]

        virtual_proj = state["proj_weight"]
        proj_heads = virtual_proj.view(
            virtual_proj.shape[0], n_head, head_dim
        )
        canonical_proj_grad_heads = canonical_proj_grad.view(
            virtual_proj.shape[0], n_head, head_dim
        )

        # Map the learned canonical projection back to the virtual
        # projection and recover the fixed value-basis block's gradient.
        virtual_proj_grad_heads = torch.einsum(
            "ohe,hde->ohd",
            canonical_proj_grad_heads,
            value_basis,
        )
        solved_tail_grad = torch.linalg.solve(
            value_basis.transpose(-1, -2),
            canonical_tail_grad,
        )
        value_prefix_grad = torch.einsum(
            "ohd,ohe->hde",
            proj_heads,
            canonical_proj_grad_heads,
        )
        value_prefix_grad = value_prefix_grad - torch.matmul(
            solved_tail_grad,
            weight_param.transpose(-1, -2),
        )
        effective_weight_grad = torch.cat(
            [value_prefix_grad, solved_tail_grad], dim=-1
        ).reshape_as(effective_weight)

        effective_reference_grad = -effective_weight_grad.sum(
            dim=1, keepdim=True
        )
        full_effective_grad = torch.cat(
            [effective_weight_grad, effective_reference_grad],
            dim=1,
        )

        constant = (
            state["full_weight"].mv(state["shift"])
            + state["full_bias"]
        )
        constant_grad = virtual_proj.transpose(0, 1).mv(
            folded_grad
        )

        virtual_proj_grad = virtual_proj_grad_heads.reshape_as(
            virtual_proj
        )
        virtual_proj_grad = virtual_proj_grad + (
            folded_grad.unsqueeze(1) * constant.unsqueeze(0)
        )
        full_proj_grad = torch.cat(
            [
                virtual_proj_grad,
                -virtual_proj_grad.sum(dim=0, keepdim=True),
            ],
            dim=0,
        )

        full_weight_grad = (
            full_effective_grad * state["scale"].unsqueeze(0)
        )
        full_weight_grad = full_weight_grad + (
            constant_grad.unsqueeze(1)
            * state["shift"].unsqueeze(0)
        )
        scale_grad = (
            full_effective_grad * state["full_weight"]
        ).sum(dim=0)
        shift_grad = state["full_weight"].transpose(0, 1).mv(
            constant_grad
        )
        full_bias_grad = constant_grad + torch.cat(
            [folded_grad, -folded_grad.sum().reshape(1)]
        )
        return (
            full_weight_grad,
            scale_grad,
            shift_grad,
            full_proj_grad,
            full_bias_grad,
        )

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        device = self.gauge_params[0].device
        total_sq = torch.zeros((), device=device)

        for param in self.all_params:
            if (
                id(param) not in self.custom_param_ids
                and param.grad is not None
            ):
                grad = param.grad.detach().float()
                total_sq.add_(grad.square().sum())

        # Include each omitted reference coordinate's full-model gradient.
        for param, axis in self.gauge_specs:
            if param.grad is not None:
                omitted_grad = -param.grad.detach().sum(
                    dim=axis, keepdim=True
                )
                total_sq.add_(omitted_grad.float().square().sum())

        # Replace composite score gradients with those of the virtual
        # full-width query, key, and query-bias parameters.
        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
        ), state in zip(self.score_specs, self.score_states):
            full_grads = self._full_score_grads(
                query_param, key_tail_param, bias_param, state
            )
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())

        # Replace folded attention gradients with gradients of the virtual
        # value, LayerNorm, full projection, and shared-bias parameters.
        for (
            weight,
            _,
            proj_weight,
            folded_bias,
        ), state in zip(
            self.attention_specs, self.attention_states
        ):
            full_grads = self._full_attention_grads(
                weight, proj_weight, folded_bias, state
            )
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())

        total_norm = total_sq.sqrt()
        clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        for param in self.all_params:
            if param.grad is not None:
                param.grad.mul_(
                    clip_coef.to(device=param.grad.device, dtype=param.grad.dtype)
                )
        return total_norm

    @torch.no_grad()
    def step(self) -> None:
        self.base.step()

        group = self.param_groups[0]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        for (param, axis), state in zip(
            self.gauge_specs, self.gauge_states
        ):
            if param.grad is None:
                continue

            grad = param.grad
            full_grad = torch.cat(
                [grad, -grad.sum(dim=axis, keepdim=True)], dim=axis
            )
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            param.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                full_grad, full_grad, value=1.0 - beta2
            )

            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(eps)
            update = exp_avg / denom
            stored_update = update.narrow(
                axis, 0, param.shape[axis]
            )
            reference_update = update.narrow(
                axis, param.shape[axis], 1
            )
            quotient_update = stored_update - reference_update
            param.add_(
                quotient_update, alpha=-lr / bias_correction1
            )

        def update_virtual(
            value, grad, exp_avg, exp_avg_sq, bias_correction1,
            bias_correction2
        ):
            value.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                grad, grad, value=1.0 - beta2
            )
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(eps)
            value.add_(
                exp_avg / denom,
                alpha=-lr / bias_correction1,
            )

        def update_quotient(
            value, full_grad, exp_avg, exp_avg_sq,
            bias_correction1, bias_correction2, axis
        ):
            value.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                full_grad, full_grad, value=1.0 - beta2
            )
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(eps)
            update = exp_avg / denom
            stored_update = update.narrow(
                axis, 0, value.shape[axis]
            )
            reference_update = update.narrow(
                axis, value.shape[axis], 1
            )
            value.add_(
                stored_update - reference_update,
                alpha=-lr / bias_correction1,
            )

        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
        ), state in zip(self.score_specs, self.score_states):
            if (
                query_param.grad is None
                and key_tail_param.grad is None
                and bias_param.grad is None
            ):
                continue

            query_grad, key_grad, bias_grad = self._full_score_grads(
                query_param, key_tail_param, bias_param, state
            )
            state["step"] += 1
            step = state["step"]
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step

            update_quotient(
                state["query_weight"],
                query_grad,
                state["exp_avg_query"],
                state["exp_avg_sq_query"],
                bias_correction1,
                bias_correction2,
                1,
            )
            update_quotient(
                state["key_weight"],
                key_grad,
                state["exp_avg_key"],
                state["exp_avg_sq_key"],
                bias_correction1,
                bias_correction2,
                1,
            )
            update_virtual(
                state["query_bias"],
                bias_grad,
                state["exp_avg_bias"],
                state["exp_avg_sq_bias"],
                bias_correction1,
                bias_correction2,
            )

            n_head, head_dim, reduced_width = query_param.shape
            query_heads = state["query_weight"].view(
                n_head, head_dim, reduced_width
            )
            key_heads = state["key_weight"].view(
                n_head, head_dim, reduced_width
            )
            bias_heads = state["query_bias"].view(
                n_head, head_dim
            )
            key_basis = key_heads[..., :head_dim]
            query_param.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            canonical_key_tail = torch.linalg.solve(
                key_basis, key_heads[..., head_dim:]
            )
            flat_key_tail = canonical_key_tail.reshape(
                -1, reduced_width - head_dim
            )
            flat_key_tail[-2, :3].zero_()
            flat_key_tail[-1].zero_()
            key_tail_param.copy_(
                torch.cat(
                    [
                        flat_key_tail[:-2].reshape(-1),
                        flat_key_tail[-2, 3:],
                    ]
                )
            )
            canonical_key_tail = flat_key_tail.view(
                n_head, head_dim, reduced_width - head_dim
            )
            projected_key_heads = torch.cat(
                [
                    key_basis,
                    torch.matmul(key_basis, canonical_key_tail),
                ],
                dim=-1,
            )
            state["key_weight"].copy_(
                projected_key_heads.reshape_as(state["key_weight"])
            )
            bias_param.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2),
                    bias_heads.unsqueeze(-1),
                ).squeeze(-1)
            )

        for (
            weight,
            _,
            proj_weight,
            folded_bias,
        ), state in zip(
            self.attention_specs, self.attention_states
        ):
            if (
                weight.grad is None
                and proj_weight.grad is None
                and folded_bias.grad is None
            ):
                continue

            (
                weight_grad,
                scale_grad,
                shift_grad,
                proj_weight_grad,
                full_bias_grad,
            ) = self._full_attention_grads(
                weight, proj_weight, folded_bias, state
            )

            state["step"] += 1
            step = state["step"]
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step

            update_virtual(
                state["full_weight"],
                weight_grad,
                state["exp_avg_weight"],
                state["exp_avg_sq_weight"],
                bias_correction1,
                bias_correction2,
            )
            update_virtual(
                state["scale"],
                scale_grad,
                state["exp_avg_scale"],
                state["exp_avg_sq_scale"],
                bias_correction1,
                bias_correction2,
            )
            update_virtual(
                state["shift"],
                shift_grad,
                state["exp_avg_shift"],
                state["exp_avg_sq_shift"],
                bias_correction1,
                bias_correction2,
            )
            update_virtual(
                state["full_bias"],
                full_bias_grad,
                state["exp_avg_bias"],
                state["exp_avg_sq_bias"],
                bias_correction1,
                bias_correction2,
            )
            update_quotient(
                state["proj_weight"],
                proj_weight_grad,
                state["exp_avg_proj"],
                state["exp_avg_sq_proj"],
                bias_correction1,
                bias_correction2,
                0,
            )

            scaled_weight = (
                state["full_weight"]
                * state["scale"].unsqueeze(0)
            )
            effective_weight = (
                scaled_weight[:, :-1] - scaled_weight[:, -1:]
            )
            n_head, head_dim, _ = weight.shape
            value_heads = effective_weight.view(
                n_head, head_dim, -1
            )
            value_basis = value_heads[..., :head_dim]
            weight.copy_(
                torch.linalg.solve(
                    value_basis, value_heads[..., head_dim:]
                )
            )

            virtual_proj_heads = state["proj_weight"].view(
                proj_weight.shape[0], n_head, head_dim
            )
            canonical_proj = torch.einsum(
                "ohd,hde->ohe",
                virtual_proj_heads,
                value_basis,
            )
            proj_weight.copy_(
                canonical_proj.reshape_as(proj_weight)
            )

            constant = (
                state["full_weight"].mv(state["shift"])
                + state["full_bias"]
            )
            folded_bias.copy_(
                state["proj_weight"].mv(constant)
            )
            folded_bias.add_(
                state["full_bias"][:-1] - state["full_bias"][-1]
            )


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
    optimizer = QuotientAdamW(
        model, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
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

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            optimizer.clip_grad_norm(train_cfg.grad_clip)
        optimizer.step()

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
    p.add_argument("--d-ff", type=int, default=1)
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
