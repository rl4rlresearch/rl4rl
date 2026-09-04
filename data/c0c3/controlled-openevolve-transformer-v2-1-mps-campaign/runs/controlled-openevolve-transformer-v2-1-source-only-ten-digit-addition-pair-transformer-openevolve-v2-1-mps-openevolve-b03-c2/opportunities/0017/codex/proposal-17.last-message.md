MECHANISM: Mean-anchored quotient bias with AdamW trajectory preservation

HYPOTHESIS: Representing each MLP output bias with seven relative coordinates and applying the difference of the corresponding eight-coordinate AdamW updates will produce a 1635-parameter model with at least 99% accuracy.

INTENDED_EDIT: Remove the final-LayerNorm-invariant uniform degree of each `fc2` bias, reconstruct a mean-anchored eight-coordinate bias, and preserve full-bias gradient clipping and AdamW quotient dynamics.

EVIDENCE: The current 1636-parameter model reached 99.92%; the earlier fixed-zero MLP-output gauge reduction collapsed to 71.82%, showing that this exact redundancy is optimization-sensitive and motivating trajectory-preserving removal.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # The final LayerNorm cancels the feature-uniform component of this
        # residual bias, so retain only its seven relative coordinates.
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        relative_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        fc2_bias = relative_bias + self.fc2.bias.mean()
        return self.drop(F.linear(hidden, self.fc2.weight, fc2_bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine
=======
def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


class QuotientAdamW(torch.optim.AdamW):
    """AdamW preserving the omitted uniform-bias coordinate's dynamics."""

    def __init__(self, params, quotient_params, **kwargs):
        self.quotient_params = list(quotient_params)
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        saved_grads = [param.grad for param in self.quotient_params]
        for param in self.quotient_params:
            param.grad = None

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
                state["quotient_step"] = 0
                state["quotient_exp_avg"] = param.new_zeros(param.numel() + 1)
                state["quotient_exp_avg_sq"] = param.new_zeros(param.numel() + 1)

            full_grad = torch.cat((grad, -grad.sum().reshape(1)))
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
                full_update[:-1] - full_update[-1],
                alpha=-step_size,
            )

        return loss


@torch.no_grad()
def clip_quotient_grad_norm_(parameters, quotient_params, max_norm: float) -> None:
    parameters = list(parameters)
    quotient_ids = {id(param) for param in quotient_params}
    total_sq = None

    for param in parameters:
        if param.grad is None:
            continue
        term = param.grad.detach().square().sum()
        if id(param) in quotient_ids:
            term = term + param.grad.detach().sum().square()
        total_sq = term if total_sq is None else total_sq + term

    if total_sq is None:
        return

    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(clip_coef)
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    quotient_params = [block.mlp.fc2.bias for block in model.blocks]
    optimizer = QuotientAdamW(
        model.parameters(),
        quotient_params,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
=======
        if train_cfg.grad_clip > 0:
            clip_quotient_grad_norm_(
                model.parameters(), quotient_params, train_cfg.grad_clip
            )
>>>>>>> REPLACE