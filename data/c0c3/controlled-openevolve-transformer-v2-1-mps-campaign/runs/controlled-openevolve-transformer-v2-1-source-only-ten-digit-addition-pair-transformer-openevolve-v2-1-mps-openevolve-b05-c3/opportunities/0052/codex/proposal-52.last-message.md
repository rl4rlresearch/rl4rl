MECHANISM: Second final-MLP output-column shift quotient

HYPOTHESIS: Extending the qualified 1,475-parameter relative-lag model’s `fc2` gauge to an adjacent hidden column will produce 1,474 learned parameters while retaining at least 99% accuracy, because either omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.

INTENDED_EDIT: Reconstruct the qualified relative-lag architecture and omit the final output-row coefficients of the last two `fc2` input columns, preserving full-shape initialization, virtual AdamW moments, weight decay, and gradient clipping.

EVIDENCE: Reference Design 3 achieved 99.94% accuracy at 1,475 parameters after quotienting one `fc2` output column; extending that same exact symmetry is better supported than adding a third attention-projection quotient, which failed at 71.64%.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedKeyLinear(nn.Linear):
    """QKV projection with one key-row coefficient fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.missing_index = d_model * d_model + d_model - 1
        self.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 1))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_index],
                self.weight.new_zeros(1),
                self.weight[self.missing_index :],
            )
        )
        return flat.view(3 * self.d_model, self.d_model)


class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection with two common-output shifts fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.missing_start = (d_model - 1) * d_model
        self.weight = nn.Parameter(torch.empty(d_model * d_model - 2))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(2),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_model)


class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with two common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 2
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 2))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(2),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 7))
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        independent_query_bias = torch.cat(
            (self.qkv.bias, self.proj.bias[-1:])
        )
        query_bias = torch.cat(
            (
                independent_query_bias,
                independent_query_bias.mean().unsqueeze(0),
            )
        )
        bias = torch.cat(
            (query_bias, self.qkv.bias.new_zeros(2 * d_model + 5))
        )
        qkv = F.linear(x, self.qkv.full_weight(), bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        lag = (positions[:, None] - positions[None, :]).clamp_min(0)
        lag_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=-1,
        )
        att = att + lag_bias[:, lag].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
=======
        y = F.linear(y, self.proj.full_weight(), self.proj.bias)
>>>>>>> REPLACE

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
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = GaugeFixedMLPProjectionLinear(d_model, d_ff)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_bias = torch.cat(
            (self.fc1.bias, self.fc1.bias.new_zeros(1))
        )
        hidden = F.linear(x, self.fc1.weight, input_bias)
        output_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        output = F.linear(
            F.gelu(hidden), self.fc2.full_weight(), output_bias
        )
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedKeyLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[d_model, -1].clone()
                full[d_model, :-1].sub_(omitted)
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_index],
                            flat[module.missing_index + 1 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, :2].clone()
                full[:, :2].sub_(omitted)
                full[-1, :2].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 2 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, -2:].clone()
                full[:, -2:].sub_(omitted)
                full[-1, -2:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 2 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.drop(self.token_emb(idx))
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
class GaugeAdamW(torch.optim.Optimizer):
    """AdamW on a last-coordinate-zero quotient."""

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
                        parameter.new_zeros(
                            *parameter.shape[:-1], 1
                        ),
                    ),
                    dim=-1,
                )
                full_value.mul_(
                    1.0 - group["lr"] * group["weight_decay"]
                )
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
    gradient = parameter.grad
    projection_gradient = gradient.clone()
    projection_gradient[-1] = -gradient[:-1].sum()
    query_gradient = gradient[-1] - projection_gradient[-1]
    return projection_gradient, query_gradient


class SharedQueryProjectionAdamW(torch.optim.Optimizer):
    """Independent AdamW dynamics for shared projection/query storage."""

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
                    state["query_exp_avg"] = torch.zeros_like(
                        query_gradient
                    )
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
                    query_gradient,
                    query_gradient,
                    value=1.0 - beta2,
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


def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    missing_start = (d_model - 1) * d_model
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(2),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_model)
    full_grad[-1, :2] = -full_grad[:-1, :2].sum(dim=0)
    return full_grad


class ProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for two attention-output shift quotients."""

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
        missing_start = self.module.missing_start
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                full_grad = full_projection_gradient(
                    parameter, d_model
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

                omitted = full_value[-1, :2].clone()
                full_value[:, :2].sub_(omitted)
                full_value[-1, :2].zero_()
                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 2 :],
                        )
                    )
                )

        return loss


def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 2
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(2),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -2:] = -full_grad[:-1, -2:].sum(dim=0)
    return full_grad


class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for two MLP common-output shift quotients."""

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
        missing_start = self.module.missing_start
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

                omitted = full_value[-1, -2:].clone()
                full_value[:, -2:].sub_(omitted)
                full_value[-1, -2:].zero_()
                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 2 :],
                        )
                    )
                )

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
    """Full-space AdamW for one LayerNorm-induced key quotient."""

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
            total_sq.add_(
                grad.sum(dim=-1).square().sum()
            )

        for key_parameter, ln_scale, d_model in key_gauges:
            if parameter is key_parameter:
                full_grad = full_key_gradient(
                    key_parameter, ln_scale, d_model
                )
                total_sq.add_(
                    full_grad[d_model, -1].float().square()
                )

        for projection_parameter, d_model in projection_gauges:
            if parameter is projection_parameter:
                full_grad = full_projection_gradient(
                    projection_parameter, d_model
                )
                total_sq.add_(
                    full_grad[-1, :2].float().square().sum()
                )

        for mlp_parameter, d_model, d_ff in mlp_projection_gauges:
            if parameter is mlp_parameter:
                full_grad = full_mlp_projection_gradient(
                    mlp_parameter, d_model, d_ff
                )
                total_sq.add_(
                    full_grad[-1, -2:].float().square().sum()
                )

    coefficient = (
        max_norm / (total_sq.sqrt() + 1e-6)
    ).clamp(max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(
                coefficient.to(dtype=parameter.grad.dtype)
            )


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    output_bias_gauge_parameters = [
        block.mlp.fc2.bias for block in model.blocks
    ]
    relative_bias_gauge_parameters = [
        block.attn.relative_bias for block in model.blocks
    ]
    last_coordinate_gauge_parameters = [
        *output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
    key_gauge_modules = [
        (block.attn.qkv, block.ln1.weight)
        for block in model.blocks
    ]
    projection_gauge_modules = [
        block.attn.proj for block in model.blocks
    ]
    mlp_projection_gauge_modules = [
        block.mlp.fc2 for block in model.blocks
    ]
    shared_query_projection_parameters = [
        block.attn.proj.bias for block in model.blocks
    ]

    excluded = {
        id(parameter)
        for parameter in last_coordinate_gauge_parameters
    }
    excluded.update(
        id(module.weight) for module, _ in key_gauge_modules
    )
    excluded.update(
        id(module.weight) for module in projection_gauge_modules
    )
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
        *projection_optimizers,
        *mlp_projection_optimizers,
        *shared_optimizers,
    ]
    key_gauges = [
        (module.weight, ln_scale, module.d_model)
        for module, ln_scale in key_gauge_modules
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
=======
        for current_optimizer in all_optimizers:
            for pg in current_optimizer.param_groups:
                pg["lr"] = lr_now
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        _, loss = model(x, y)
        for current_optimizer in all_optimizers:
            current_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_gauges(
                model,
                last_coordinate_gauge_parameters,
                key_gauges,
                projection_gauges,
                mlp_projection_gauges,
                shared_query_projection_parameters,
                train_cfg.grad_clip,
            )
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for projection_optimizer in projection_optimizers:
            projection_optimizer.step()
        for mlp_projection_optimizer in mlp_projection_optimizers:
            mlp_projection_optimizer.step()
        for shared_optimizer in shared_optimizers:
            shared_optimizer.step()
        for gauge_optimizer in last_coordinate_optimizers:
            gauge_optimizer.step()
        optimizer.step()
>>>>>>> REPLACE