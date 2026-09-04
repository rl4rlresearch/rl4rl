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


class GaugeAdamW(torch.optim.Optimizer):
    """AdamW on an embedding quotient with one virtual gauge coordinate."""

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(
            [parameter],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                reduced_grad = parameter.grad
                full_grad = torch.cat(
                    (
                        reduced_grad,
                        -reduced_grad.sum(dim=-1, keepdim=True),
                    ),
                    dim=-1,
                )
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(full_grad)
                    state["exp_avg_sq"] = torch.zeros_like(full_grad)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )

                step = state["step"]
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(group["eps"])

                full_value = torch.cat(
                    (
                        parameter,
                        parameter.new_zeros(*parameter.shape[:-1], 1),
                    ),
                    dim=-1,
                )
                full_value.mul_(1.0 - group["lr"] * group["weight_decay"])
                full_value.addcdiv_(
                    exp_avg,
                    denom,
                    value=-group["lr"] / bias_correction1,
                )
                parameter.copy_(
                    full_value[..., :-1] - full_value[..., -1:]
                )

        return loss


def shared_query_projection_gradients(
    parameter: torch.nn.Parameter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recover independent full projection and scalar query gradients."""
    gradient = parameter.grad
    projection_gradient = gradient.clone()
    projection_gradient[-1] = -gradient[:-1].sum()
    query_gradient = gradient[-1] - projection_gradient[-1]
    return projection_gradient, query_gradient


class SharedQueryProjectionAdamW(torch.optim.Optimizer):
    """Independent AdamW dynamics in a shared projection-shift coordinate."""

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(
            [parameter],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                projection_gradient, query_gradient = (
                    shared_query_projection_gradients(parameter)
                )
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["projection_exp_avg"] = torch.zeros_like(
                        projection_gradient
                    )
                    state["projection_exp_avg_sq"] = torch.zeros_like(
                        projection_gradient
                    )
                    state["query_exp_avg"] = torch.zeros_like(query_gradient)
                    state["query_exp_avg_sq"] = torch.zeros_like(
                        query_gradient
                    )

                state["step"] += 1
                projection_exp_avg = state["projection_exp_avg"]
                projection_exp_avg_sq = state["projection_exp_avg_sq"]
                query_exp_avg = state["query_exp_avg"]
                query_exp_avg_sq = state["query_exp_avg_sq"]

                projection_exp_avg.mul_(beta1).add_(
                    projection_gradient, alpha=1.0 - beta1
                )
                projection_exp_avg_sq.mul_(beta2).addcmul_(
                    projection_gradient,
                    projection_gradient,
                    value=1.0 - beta2,
                )
                query_exp_avg.mul_(beta1).add_(
                    query_gradient, alpha=1.0 - beta1
                )
                query_exp_avg_sq.mul_(beta2).addcmul_(
                    query_gradient, query_gradient, value=1.0 - beta2
                )

                step = state["step"]
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                projection_denom = projection_exp_avg_sq.sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(group["eps"])
                query_denom = query_exp_avg_sq.sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(group["eps"])

                projection_value = parameter.clone()
                query_value = parameter[-1].clone()
                decay = 1.0 - group["lr"] * group["weight_decay"]
                projection_value.mul_(decay)
                query_value.mul_(decay)
                projection_value.addcdiv_(
                    projection_exp_avg,
                    projection_denom,
                    value=-group["lr"] / bias_correction1,
                )
                query_value.addcdiv_(
                    query_exp_avg,
                    query_denom,
                    value=-group["lr"] / bias_correction1,
                )

                relative_projection = (
                    projection_value[:-1] - projection_value[-1]
                )
                parameter[:-1].copy_(
                    relative_projection + query_value
                )
                parameter[-1].copy_(query_value)

        return loss


def layer_norm_scale(ln_module: torch.nn.Module) -> torch.Tensor:
    if hasattr(ln_module, "full_weight"):
        return ln_module.full_weight()
    return ln_module.weight


def full_input_gradient(
    parameter: torch.nn.Parameter,
    ln_module: torch.nn.Module,
) -> torch.Tensor:
    full_grad = torch.cat(
        (
            parameter.grad,
            parameter.grad.new_zeros(parameter.shape[0], 1),
        ),
        dim=-1,
    )
    gamma = layer_norm_scale(ln_module).detach()
    full_grad[:, -1] = -gamma[-1] * (
        full_grad[:, :-1] / gamma[:-1]
    ).sum(dim=-1)
    return full_grad


class InputGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for LayerNorm-nullspace input-map quotients."""

    def __init__(
        self,
        module: torch.nn.Module,
        ln_module: torch.nn.Module,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.module = module
        self.ln_module = ln_module
        super().__init__(
            [module.weight],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                full_grad = full_input_gradient(
                    parameter, self.ln_module
                )
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(full_grad)
                    state["exp_avg_sq"] = torch.zeros_like(full_grad)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(
                    full_grad, alpha=1.0 - beta1
                )
                exp_avg_sq.mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )

                step = state["step"]
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(group["eps"])

                full_value = self.module.full_weight()
                full_value.mul_(
                    1.0 - group["lr"] * group["weight_decay"]
                )
                full_value.addcdiv_(
                    exp_avg,
                    denom,
                    value=-group["lr"] / bias_correction1,
                )

                gamma = layer_norm_scale(self.ln_module).detach()
                omitted = full_value[:, -1:].clone()
                full_value[:, :-1].sub_(
                    omitted * gamma[-1] / gamma[:-1]
                )
                full_value[:, -1].zero_()
                parameter.copy_(full_value[:, :-1])

        return loss


def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(3),
            parameter.grad,
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad


class ProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for all attention-output shift quotients."""

    def __init__(
        self,
        module: torch.nn.Module,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.module = module
        super().__init__(
            [module.weight],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        d_model = self.module.d_model
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                full_grad = full_projection_gradient(parameter, d_model)
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(full_grad)
                    state["exp_avg_sq"] = torch.zeros_like(full_grad)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )

                step = state["step"]
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(group["eps"])

                full_value = self.module.full_weight()
                full_value.mul_(
                    1.0 - group["lr"] * group["weight_decay"]
                )
                full_value.addcdiv_(
                    exp_avg,
                    denom,
                    value=-group["lr"] / bias_correction1,
                )

                omitted = full_value[-1].clone()
                full_value.sub_(omitted)
                full_value[-1].zero_()

                anchors = full_value[0, :3].clone()
                scale = self.module.fixed_anchor / anchors
                full_value[:, :3].mul_(scale)
                self.module._value_source.weight[:3].div_(
                    scale[:, None]
                )
                parameter.copy_(full_value[:-1].reshape(-1)[3:])

        return loss


def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    full_grad = torch.cat(
        (
            parameter.grad,
            parameter.grad.new_zeros(d_ff),
        )
    ).view(d_model, d_ff)
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad


class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for all MLP common-output shift quotients."""

    def __init__(
        self,
        module: torch.nn.Module,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.module = module
        super().__init__(
            [module.weight],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        d_model = self.module.d_model
        d_ff = self.module.d_ff
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                full_grad = full_mlp_projection_gradient(
                    parameter, d_model, d_ff
                )
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(full_grad)
                    state["exp_avg_sq"] = torch.zeros_like(full_grad)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )

                step = state["step"]
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(group["eps"])

                full_value = self.module.full_weight()
                full_value.mul_(
                    1.0 - group["lr"] * group["weight_decay"]
                )
                full_value.addcdiv_(
                    exp_avg,
                    denom,
                    value=-group["lr"] / bias_correction1,
                )

                omitted = full_value[-1].clone()
                full_value.sub_(omitted)
                full_value[-1].zero_()
                parameter.copy_(full_value[:-1].reshape(-1))

        return loss


def full_key_gradient(
    parameter: torch.nn.Parameter,
    ln_scale: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    missing_index = d_model * d_model + d_model - 1
    full_grad = torch.cat(
        (
            parameter.grad[:missing_index],
            parameter.grad.new_zeros(1),
            parameter.grad[missing_index:],
        )
    ).view(3 * d_model, d_model)
    gamma = ln_scale.detach()
    full_grad[d_model, -1] = -gamma[-1] * (
        full_grad[d_model, :-1] / gamma[:-1]
    ).sum()
    return full_grad


class KeyGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for one LayerNorm-induced key-weight quotient."""

    def __init__(
        self,
        module: torch.nn.Module,
        ln_scale: torch.nn.Parameter,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.module = module
        self.ln_scale = ln_scale
        super().__init__(
            [module.weight],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        d_model = self.module.d_model
        missing_index = self.module.missing_index
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                full_grad = full_key_gradient(
                    parameter, self.ln_scale, d_model
                )
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(full_grad)
                    state["exp_avg_sq"] = torch.zeros_like(full_grad)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )

                step = state["step"]
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(group["eps"])

                full_value = self.module.full_weight()
                full_value.mul_(
                    1.0 - group["lr"] * group["weight_decay"]
                )
                full_value.addcdiv_(
                    exp_avg,
                    denom,
                    value=-group["lr"] / bias_correction1,
                )

                gamma = self.ln_scale.detach()
                key_row = full_value[d_model]
                omitted = key_row[-1].clone()
                key_row[:-1].sub_(
                    omitted * gamma[-1] / gamma[:-1]
                )
                key_row[-1].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_index],
                            flat[missing_index + 1 :],
                        )
                    )
                )

        return loss


@torch.no_grad()
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    last_coordinate_gauges: List[torch.nn.Parameter],
    key_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Parameter, int]
    ],
    input_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Module]
    ],
    projection_gauges: List[Tuple[torch.nn.Parameter, int]],
    mlp_projection_gauges: List[
        Tuple[torch.nn.Parameter, int, int]
    ],
    shared_parameters: List[torch.nn.Parameter],
    max_norm: float,
) -> None:
    reference_parameter = next(model.parameters())
    total_sq = torch.zeros(
        (), device=reference_parameter.device, dtype=torch.float32
    )
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()

        if any(
            parameter is shared_parameter
            for shared_parameter in shared_parameters
        ):
            projection_gradient, query_gradient = (
                shared_query_projection_gradients(parameter)
            )
            total_sq.add_(
                projection_gradient.detach().float().square().sum()
            )
            total_sq.add_(
                query_gradient.detach().float().square()
            )
        else:
            total_sq.add_(grad.square().sum())

        if any(
            parameter is gauge_parameter
            for gauge_parameter in last_coordinate_gauges
        ):
            total_sq.add_(grad.sum(dim=-1).square().sum())

        for key_parameter, ln_scale, d_model in key_gauges:
            if parameter is key_parameter:
                full_grad = full_key_gradient(
                    key_parameter, ln_scale, d_model
                )
                total_sq.add_(
                    full_grad[d_model, -1].float().square()
                )

        for input_parameter, ln_module in input_gauges:
            if parameter is input_parameter:
                full_grad = full_input_gradient(
                    input_parameter, ln_module
                )
                total_sq.add_(
                    full_grad[:, -1].float().square().sum()
                )

        for projection_parameter, d_model in projection_gauges:
            if parameter is projection_parameter:
                full_grad = full_projection_gradient(
                    projection_parameter, d_model
                )
                total_sq.add_(
                    full_grad[-1].float().square().sum()
                )

        for mlp_parameter, d_model, d_ff in mlp_projection_gauges:
            if parameter is mlp_parameter:
                full_grad = full_mlp_projection_gradient(
                    mlp_parameter, d_model, d_ff
                )
                total_sq.add_(
                    full_grad[-1].float().square().sum()
                )

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(
                coefficient.to(dtype=parameter.grad.dtype)
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

    embedding_gauge_parameters = [
        model.token_emb.weight
    ]
    output_bias_gauge_parameters = [
        block.mlp.fc2.bias for block in model.blocks
    ]
    attention_output_bias_gauge_parameters = [
        block.attn.proj.bias for block in model.blocks
    ]
    relative_bias_gauge_parameters = [
        block.attn.relative_bias for block in model.blocks
    ]
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
    key_gauge_modules = []
    input_gauge_modules = [
        pair
        for block in model.blocks
        for pair in (
            (block.attn.value, block.ln1),
            (block.mlp.fc1, block.ln2),
        )
    ]
    projection_gauge_modules = [
        block.attn.proj for block in model.blocks
    ]
    mlp_projection_gauge_modules = [
        block.mlp.fc2 for block in model.blocks
    ]
    shared_query_projection_parameters = []
    excluded = {
        id(parameter)
        for parameter in last_coordinate_gauge_parameters
    }
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
    excluded.update(
        id(module.weight) for module, _ in input_gauge_modules
    )
    excluded.update(id(module.weight) for module in projection_gauge_modules)
    excluded.update(
        id(module.weight) for module in mlp_projection_gauge_modules
    )
    excluded.update(
        id(parameter)
        for parameter in shared_query_projection_parameters
    )
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in excluded
    ]

    optimizer = torch.optim.AdamW(
        regular_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    last_coordinate_optimizers = [
        GaugeAdamW(
            parameter,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for parameter in last_coordinate_gauge_parameters
    ]
    key_optimizers = [
        KeyGaugeAdamW(
            module,
            ln_scale,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for module, ln_scale in key_gauge_modules
    ]
    input_optimizers = [
        InputGaugeAdamW(
            module,
            ln_module,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for module, ln_module in input_gauge_modules
    ]
    projection_optimizers = [
        ProjectionGaugeAdamW(
            module,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for module in projection_gauge_modules
    ]
    mlp_projection_optimizers = [
        MLPProjectionGaugeAdamW(
            module,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for module in mlp_projection_gauge_modules
    ]
    shared_optimizers = [
        SharedQueryProjectionAdamW(
            parameter,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for parameter in shared_query_projection_parameters
    ]
    all_optimizers = [
        optimizer,
        *last_coordinate_optimizers,
        *key_optimizers,
        *input_optimizers,
        *projection_optimizers,
        *mlp_projection_optimizers,
        *shared_optimizers,
    ]
    key_gauges = [
        (module.weight, ln_scale, module.d_model)
        for module, ln_scale in key_gauge_modules
    ]
    input_gauges = [
        (module.weight, ln_module)
        for module, ln_module in input_gauge_modules
    ]
    projection_gauges = [
        (module.weight, module.d_model)
        for module in projection_gauge_modules
    ]
    mlp_projection_gauges = [
        (module.weight, module.d_model, module.d_ff)
        for module in mlp_projection_gauge_modules
    ]

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
        for current_optimizer in all_optimizers:
            for pg in current_optimizer.param_groups:
                pg["lr"] = lr_now

        _, loss = model(x, y)
        for current_optimizer in all_optimizers:
            current_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_gauges(
                model,
                last_coordinate_gauge_parameters,
                key_gauges,
                input_gauges,
                projection_gauges,
                mlp_projection_gauges,
                shared_query_projection_parameters,
                train_cfg.grad_clip,
            )
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for input_optimizer in input_optimizers:
            input_optimizer.step()
        for projection_optimizer in projection_optimizers:
            projection_optimizer.step()
        for mlp_projection_optimizer in mlp_projection_optimizers:
            mlp_projection_optimizer.step()
        for shared_optimizer in shared_optimizers:
            shared_optimizer.step()
        for gauge_optimizer in last_coordinate_optimizers:
            gauge_optimizer.step()
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
