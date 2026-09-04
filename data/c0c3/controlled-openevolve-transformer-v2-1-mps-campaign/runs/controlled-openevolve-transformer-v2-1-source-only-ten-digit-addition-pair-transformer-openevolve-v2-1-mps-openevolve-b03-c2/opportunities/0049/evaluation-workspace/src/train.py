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


def reconstruct_input_weight(param):
    relative = torch.cat(
        (
            param,
            param.new_zeros((param.size(0), 1)),
        ),
        dim=-1,
    )
    return relative + param.mean(dim=-1, keepdim=True)


def reconstruct_attention_output_weight(
    weight, head_weight, last_weight, head_dim
):
    head_relative = torch.cat(
        (head_weight, head_weight.new_zeros(1))
    )
    last_relative = torch.cat(
        (last_weight, last_weight.new_zeros(1))
    )
    split = head_dim - 1
    rows = torch.cat(
        (
            weight[:split],
            head_relative.unsqueeze(0),
            weight[split:],
            last_relative.unsqueeze(0),
        ),
        dim=0,
    )
    relative = torch.cat(
        (rows, rows.new_zeros((rows.size(0), 1))),
        dim=-1,
    )
    return (
        relative + rows.mean(dim=-1, keepdim=True)
    ).transpose(0, 1)


class QuotientAdamW(torch.optim.AdamW):
    """AdamW preserving folded and omitted full-coordinate dynamics."""

    def __init__(
        self,
        params,
        quotient_params,
        value_bias_specs=(),
        factor_params=(),
        embedding_specs=(),
        **kwargs,
    ):
        self.quotient_params = list(quotient_params)
        self.value_bias_specs = list(value_bias_specs)
        self.factor_params = list(factor_params)
        self.embedding_specs = list(embedding_specs)
        super().__init__(params, **kwargs)

    def _factor_state(self, param):
        state = self.state[param]
        if "factor_weight" not in state:
            state["factor_step"] = 0
            state["factor_weight"] = reconstruct_input_weight(
                param.detach()
            ).clone()
            state["factor_scale"] = param.new_ones(param.size(1) + 1)
            state["factor_weight_exp_avg"] = torch.zeros_like(
                state["factor_weight"]
            )
            state["factor_weight_exp_avg_sq"] = torch.zeros_like(
                state["factor_weight"]
            )
            state["factor_scale_exp_avg"] = param.new_zeros(
                param.size(1) + 1
            )
            state["factor_scale_exp_avg_sq"] = param.new_zeros(
                param.size(1) + 1
            )
        return state

    @torch.no_grad()
    def factor_grad_sq(self, param):
        state = self._factor_state(param)
        grad = param.grad.detach()
        full_grad = torch.cat(
            (grad, -grad.sum(dim=-1, keepdim=True)),
            dim=-1,
        )
        weight_grad = (
            full_grad * state["factor_scale"].unsqueeze(0)
        )
        scale_grad = (
            full_grad * state["factor_weight"]
        ).sum(dim=0)
        return weight_grad.square().sum() + scale_grad.square().sum()

    @staticmethod
    def _embedding_full_grads(token_grad, pos_grad):
        full_pos_grad = torch.cat(
            (
                pos_grad,
                -pos_grad.sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        final_token_grad = (
            full_pos_grad.sum(dim=0) - token_grad.sum(dim=0)
        )
        full_token_grad = torch.cat(
            (token_grad, final_token_grad.unsqueeze(0)),
            dim=0,
        )
        return full_token_grad, full_pos_grad

    @torch.no_grad()
    def embedding_grad_sq(self, token_param, pos_param):
        if token_param.grad is None or pos_param.grad is None:
            return None
        full_token_grad, full_pos_grad = self._embedding_full_grads(
            token_param.grad.detach(),
            pos_param.grad.detach(),
        )
        return (
            full_token_grad.square().sum()
            + full_pos_grad.square().sum()
        )

    @torch.no_grad()
    def step(self, closure=None):
        value_bias_grads = []
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
        ) in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                full_proj_grad = torch.cat(
                    (
                        proj_bias.grad.detach(),
                        -proj_bias.grad.detach().sum().view(1),
                    )
                )
                full_proj_weight = (
                    reconstruct_attention_output_weight(
                        proj_weight.detach(),
                        proj_head_weight.detach(),
                        proj_last_weight.detach(),
                        head_dim,
                    )
                )
                grad = (
                    full_proj_weight
                    * full_proj_grad.unsqueeze(1)
                ).sum(dim=0)
            value_bias_grads.append(grad)

        saved_grads = [param.grad for param in self.quotient_params]
        saved_factor_grads = [
            param.grad for param in self.factor_params
        ]
        saved_embedding_grads = [
            (token_param.grad, pos_param.grad)
            for token_param, pos_param in self.embedding_specs
        ]
        for param in self.quotient_params:
            param.grad = None
        for param in self.factor_params:
            param.grad = None
        for token_param, pos_param in self.embedding_specs:
            token_param.grad = None
            pos_param.grad = None

        loss = super().step(closure)

        for param, grad in zip(self.quotient_params, saved_grads):
            param.grad = grad
            if grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(candidate is param for candidate in group["params"])
            )
            state = self.state[param]
            if "quotient_step" not in state:
                full_shape = list(param.shape)
                full_shape[-1] += 1
                state["quotient_step"] = 0
                state["quotient_exp_avg"] = param.new_zeros(full_shape)
                state["quotient_exp_avg_sq"] = param.new_zeros(full_shape)

            full_grad = torch.cat(
                (grad, -grad.sum(dim=-1, keepdim=True)),
                dim=-1,
            )
            if group["maximize"]:
                full_grad = -full_grad

            state["quotient_step"] += 1
            step = state["quotient_step"]
            beta1, beta2 = group["betas"]
            exp_avg = state["quotient_exp_avg"]
            exp_avg_sq = state["quotient_exp_avg_sq"]

            exp_avg.lerp_(full_grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                full_grad, full_grad, value=1.0 - beta2
            )

            lr = group["lr"]
            param.mul_(1.0 - lr * group["weight_decay"])
            step_size = lr / (1.0 - beta1 ** step)
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(1.0 - beta2 ** step)
            ).add_(group["eps"])
            full_update = exp_avg / denom
            param.add_(
                full_update[..., :-1] - full_update[..., -1:],
                alpha=-step_size,
            )

        # Update full token and positional embeddings independently, then
        # return to their coupled common-offset gauge.
        for (
            token_param,
            pos_param,
        ), (
            token_grad,
            pos_grad,
        ) in zip(self.embedding_specs, saved_embedding_grads):
            token_param.grad = token_grad
            pos_param.grad = pos_grad
            if token_grad is None or pos_grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(
                    candidate is token_param
                    for candidate in group["params"]
                )
            )
            state = self.state[token_param]
            full_token_grad, full_pos_grad = (
                self._embedding_full_grads(
                    token_grad.detach(),
                    pos_grad.detach(),
                )
            )
            if group["maximize"]:
                full_token_grad = -full_token_grad
                full_pos_grad = -full_pos_grad

            if "embedding_step" not in state:
                state["embedding_step"] = 0
                state["token_exp_avg"] = torch.zeros_like(
                    full_token_grad
                )
                state["token_exp_avg_sq"] = torch.zeros_like(
                    full_token_grad
                )
                state["pos_exp_avg"] = torch.zeros_like(
                    full_pos_grad
                )
                state["pos_exp_avg_sq"] = torch.zeros_like(
                    full_pos_grad
                )

            state["embedding_step"] += 1
            step = state["embedding_step"]
            beta1, beta2 = group["betas"]
            token_exp_avg = state["token_exp_avg"]
            token_exp_avg_sq = state["token_exp_avg_sq"]
            pos_exp_avg = state["pos_exp_avg"]
            pos_exp_avg_sq = state["pos_exp_avg_sq"]

            token_exp_avg.lerp_(full_token_grad, 1.0 - beta1)
            token_exp_avg_sq.mul_(beta2).addcmul_(
                full_token_grad,
                full_token_grad,
                value=1.0 - beta2,
            )
            pos_exp_avg.lerp_(full_pos_grad, 1.0 - beta1)
            pos_exp_avg_sq.mul_(beta2).addcmul_(
                full_pos_grad,
                full_pos_grad,
                value=1.0 - beta2,
            )

            lr = group["lr"]
            decay = 1.0 - lr * group["weight_decay"]
            token_param.mul_(decay)
            pos_param.mul_(decay)
            step_size = lr / (1.0 - beta1 ** step)
            bias_correction2 = math.sqrt(1.0 - beta2 ** step)
            token_update = token_exp_avg / (
                token_exp_avg_sq.sqrt().div_(
                    bias_correction2
                ).add_(group["eps"])
            )
            pos_update = pos_exp_avg / (
                pos_exp_avg_sq.sqrt().div_(
                    bias_correction2
                ).add_(group["eps"])
            )

            token_param.add_(
                token_update[:-1] - token_update[-1:],
                alpha=-step_size,
            )
            shifted_pos_update = (
                pos_update + token_update[-1].unsqueeze(0)
            )
            pos_param.add_(
                shifted_pos_update[:, :-1]
                - shifted_pos_update[:, -1:],
                alpha=-step_size,
            )

        # Reproduce AdamW on each downstream weight and its omitted
        # LayerNorm scale, then store their sufficient columnwise product.
        for param, grad in zip(
            self.factor_params, saved_factor_grads
        ):
            param.grad = grad
            if grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(candidate is param for candidate in group["params"])
            )
            state = self._factor_state(param)
            factor_weight = state["factor_weight"]
            factor_scale = state["factor_scale"]
            full_grad = torch.cat(
                (grad, -grad.sum(dim=-1, keepdim=True)),
                dim=-1,
            )
            weight_grad = full_grad * factor_scale.unsqueeze(0)
            scale_grad = (full_grad * factor_weight).sum(dim=0)
            if group["maximize"]:
                weight_grad = -weight_grad
                scale_grad = -scale_grad

            state["factor_step"] += 1
            step = state["factor_step"]
            beta1, beta2 = group["betas"]
            weight_exp_avg = state["factor_weight_exp_avg"]
            weight_exp_avg_sq = state["factor_weight_exp_avg_sq"]
            scale_exp_avg = state["factor_scale_exp_avg"]
            scale_exp_avg_sq = state["factor_scale_exp_avg_sq"]

            weight_exp_avg.lerp_(weight_grad, 1.0 - beta1)
            weight_exp_avg_sq.mul_(beta2).addcmul_(
                weight_grad, weight_grad, value=1.0 - beta2
            )
            scale_exp_avg.lerp_(scale_grad, 1.0 - beta1)
            scale_exp_avg_sq.mul_(beta2).addcmul_(
                scale_grad, scale_grad, value=1.0 - beta2
            )

            lr = group["lr"]
            decay = 1.0 - lr * group["weight_decay"]
            factor_weight.mul_(decay)
            factor_scale.mul_(decay)
            step_size = lr / (1.0 - beta1 ** step)
            bias_correction2 = math.sqrt(1.0 - beta2 ** step)
            weight_denom = weight_exp_avg_sq.sqrt().div_(
                bias_correction2
            ).add_(group["eps"])
            scale_denom = scale_exp_avg_sq.sqrt().div_(
                bias_correction2
            ).add_(group["eps"])
            factor_weight.addcdiv_(
                weight_exp_avg,
                weight_denom,
                value=-step_size,
            )
            factor_scale.addcdiv_(
                scale_exp_avg,
                scale_denom,
                value=-step_size,
            )
            full_product = (
                factor_weight * factor_scale.unsqueeze(0)
            )
            param.copy_(
                full_product[:, :-1] - full_product[:, -1:]
            )

        # Update the omitted value-bias coordinate in full-coordinate AdamW,
        # then immediately return to the zero-coordinate gauge by folding its
        # effect into the already-updated attention projection bias.
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
            if grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(candidate is qkv_bias for candidate in group["params"])
            )
            if group["maximize"]:
                grad = -grad

            state = self.state[qkv_bias]
            if "value_quotient_step" not in state:
                state["value_quotient_step"] = 0
                state["value_quotient_exp_avg"] = qkv_bias.new_zeros(
                    grad.shape
                )
                state["value_quotient_exp_avg_sq"] = qkv_bias.new_zeros(
                    grad.shape
                )

            state["value_quotient_step"] += 1
            step = state["value_quotient_step"]
            beta1, beta2 = group["betas"]
            exp_avg = state["value_quotient_exp_avg"]
            exp_avg_sq = state["value_quotient_exp_avg_sq"]

            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                grad, grad, value=1.0 - beta2
            )

            step_size = group["lr"] / (1.0 - beta1 ** step)
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(1.0 - beta2 ** step)
            ).add_(group["eps"])
            omitted_value = -step_size * exp_avg / denom
            full_delta = (
                reconstruct_attention_output_weight(
                    proj_weight,
                    proj_head_weight,
                    proj_last_weight,
                    head_dim,
                )
                @ omitted_value
            )
            proj_bias.add_(
                full_delta[:-1] - full_delta[-1]
            )

        return loss


@torch.no_grad()
def clip_quotient_grad_norm_(
    parameters,
    quotient_params,
    value_bias_specs,
    max_norm: float,
    factor_optimizer=None,
) -> None:
    parameters = list(parameters)
    quotient_ids = {id(param) for param in quotient_params}
    factor_ids = (
        {id(param) for param in factor_optimizer.factor_params}
        if factor_optimizer is not None
        else set()
    )
    embedding_ids = (
        {
            id(param)
            for spec in factor_optimizer.embedding_specs
            for param in spec
        }
        if factor_optimizer is not None
        else set()
    )
    total_sq = None

    for param in parameters:
        if param.grad is None or id(param) in embedding_ids:
            continue
        grad = param.grad.detach()
        if id(param) in factor_ids:
            term = factor_optimizer.factor_grad_sq(param)
        else:
            term = grad.square().sum()
            if id(param) in quotient_ids:
                term = term + grad.sum(dim=-1).square().sum()
        total_sq = term if total_sq is None else total_sq + term

    if factor_optimizer is not None:
        for token_param, pos_param in factor_optimizer.embedding_specs:
            term = factor_optimizer.embedding_grad_sq(
                token_param, pos_param
            )
            if term is not None:
                total_sq = (
                    term if total_sq is None else total_sq + term
                )

    for (
        qkv_bias,
        proj_weight,
        proj_head_weight,
        proj_last_weight,
        proj_bias,
        head_dim,
    ) in value_bias_specs:
        if proj_bias.grad is None:
            continue
        full_proj_grad = torch.cat(
            (
                proj_bias.grad.detach(),
                -proj_bias.grad.detach().sum().view(1),
            )
        )
        full_proj_weight = reconstruct_attention_output_weight(
            proj_weight.detach(),
            proj_head_weight.detach(),
            proj_last_weight.detach(),
            head_dim,
        )
        omitted_grad = (
            full_proj_weight
            * full_proj_grad.unsqueeze(1)
        ).sum(dim=0)
        term = omitted_grad.square().sum()
        total_sq = term if total_sq is None else total_sq + term

    if total_sq is None:
        return

    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(clip_coef)


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
    quotient_params = [
        block.attn.proj.bias for block in model.blocks
    ] + [
        block.attn.proj.weight for block in model.blocks
    ] + [
        block.mlp.fc2.bias for block in model.blocks
    ] + [
        block.mlp.fc2.weight for block in model.blocks
    ]
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_head_weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
        )
        for block in model.blocks
    ]
    factor_params = [
        block.attn.qkv.weight for block in model.blocks
    ] + [
        block.mlp.fc1.weight for block in model.blocks
    ]
    embedding_specs = [
        (model.token_emb.weight, model.pos_emb.weight)
    ]
    optimizer = QuotientAdamW(
        model.parameters(),
        quotient_params,
        value_bias_specs,
        factor_params,
        embedding_specs=embedding_specs,
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

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_quotient_grad_norm_(
                model.parameters(),
                quotient_params,
                value_bias_specs,
                train_cfg.grad_clip,
                optimizer,
            )
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
