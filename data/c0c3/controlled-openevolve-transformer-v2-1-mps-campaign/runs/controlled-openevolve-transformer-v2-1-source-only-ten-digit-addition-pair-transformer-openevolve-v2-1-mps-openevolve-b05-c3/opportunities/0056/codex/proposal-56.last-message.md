MECHANISM: Sixth final-MLP common-output column gauge with learned relative-lag attention

HYPOTHESIS: Extending the verified five-column `fc2` gauge to a sixth adjacent hidden column will produce 1,470 learned parameters and retain at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.

INTENDED_EDIT: Replace absolute positional embeddings with the qualified learned relative-lag biases and gauge-fix the last six `fc2` input columns, preserving full-space initialization, AdamW moments, weight decay, and gauge-aware clipping.

EVIDENCE: The five-column quotient achieved 99.95% accuracy at 1,471 parameters after the one-through-four-column variants all exceeded 99%; this applies the same repeatedly verified exact symmetry to one additional adjacent column.

<<<<<<< SEARCH
        return flat.view(self.d_model, self.d_model)


class CausalSelfAttention(nn.Module):
=======
        return flat.view(self.d_model, self.d_model)


class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with six common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 6
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 6))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(6),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
=======
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
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
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        output_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        return self.drop(output)
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
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        output_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        output = F.linear(
            F.gelu(hidden), self.fc2.full_weight(), output_bias
        )
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, -6:].clone()
                full[:, -6:].sub_(omitted)
                full[-1, -6:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 6 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.drop(self.token_emb(idx))
>>>>>>> REPLACE

<<<<<<< SEARCH
        return loss


def full_key_gradient(
=======
        return loss


def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 6
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(6),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -6:] = -full_grad[:-1, -6:].sum(dim=0)
    return full_grad


class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for six MLP common-output shift quotients."""

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

                omitted = full_value[-1, -6:].clone()
                full_value[:, -6:].sub_(omitted)
                full_value[-1, -6:].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 6 :],
                        )
                    )
                )

        return loss


def full_key_gradient(
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
    output_bias_gauges: List[torch.nn.Parameter],
    key_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Parameter, int]
    ],
    projection_gauges: List[Tuple[torch.nn.Parameter, int]],
    shared_parameters: List[torch.nn.Parameter],
    max_norm: float,
) -> None:
    total_sq = torch.zeros(
        (), device=position_parameter.device, dtype=torch.float32
    )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        if parameter is position_parameter:
            total_sq.add_(grad.sum(dim=-1).square().sum())

        if any(
            parameter is output_bias
            for output_bias in output_bias_gauges
        ):
            total_sq.add_(grad.sum().square())
=======
        if any(
            parameter is gauge_parameter
            for gauge_parameter in last_coordinate_gauges
        ):
            total_sq.add_(grad.sum(dim=-1).square().sum())
>>>>>>> REPLACE

<<<<<<< SEARCH
        for projection_parameter, d_model in projection_gauges:
            if parameter is projection_parameter:
                full_grad = full_projection_gradient(
                    projection_parameter, d_model
                )
                total_sq.add_(
                    full_grad[-1, :2].float().square().sum()
                )

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
=======
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
                    full_grad[-1, -6:].float().square().sum()
                )

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    position_parameter = model.pos_emb.weight
    output_bias_gauge_parameters = [
        block.mlp.fc2.bias for block in model.blocks
    ]
    key_gauge_modules = [
        (block.attn.qkv, block.ln1.weight)
        for block in model.blocks
    ]
    projection_gauge_modules = [
        block.attn.proj for block in model.blocks
    ]
    shared_query_projection_parameters = [
        block.attn.proj.bias for block in model.blocks
    ]
    excluded = {id(position_parameter)}
    excluded.update(
        id(parameter) for parameter in output_bias_gauge_parameters
    )
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
    excluded.update(id(module.weight) for module in projection_gauge_modules)
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
    position_optimizer = GaugeAdamW(
        position_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    output_bias_optimizers = [
        GaugeAdamW(
            parameter,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for parameter in output_bias_gauge_parameters
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
        position_optimizer,
        *output_bias_optimizers,
        *key_optimizers,
        *projection_optimizers,
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
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_grad_norm_with_gauges(
                model,
                position_parameter,
                output_bias_gauge_parameters,
                key_gauges,
                projection_gauges,
                shared_query_projection_parameters,
                train_cfg.grad_clip,
            )
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for projection_optimizer in projection_optimizers:
            projection_optimizer.step()
        for shared_optimizer in shared_optimizers:
            shared_optimizer.step()
        for output_bias_optimizer in output_bias_optimizers:
            output_bias_optimizer.step()
        position_optimizer.step()
        optimizer.step()
=======
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