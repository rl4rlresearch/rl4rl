MECHANISM: LayerNorm-affine quotient for one MLP input row

HYPOTHESIS: Gauge-fixing unit 10’s final `fc1` input coefficient while retaining its learned bias will reduce the verified model to 1598 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Represent one `fc1` row with seven learned coefficients, reconstruct its omitted LayerNorm-gauge gradient, and optimize its weight and bias with gauge-aware AdamW and clipping.

EVIDENCE: The current single-key-row quotient achieved 99.88% at 1599 parameters, while removing unit 10’s bias nearly passed at 98.57%; the previously proposed unit-10 weight quotient was not verified, so preserving its threshold while removing an exact affine redundancy remains the most informative test.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedMLPLinear(nn.Linear):
    """MLP input projection with one LayerNorm-redundant coefficient omitted."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_model, d_ff)
        self.d_model = d_model
        self.d_ff = d_ff
        self.gauge_row = d_ff - 2
        self.missing_index = (
            self.gauge_row * d_model + d_model - 1
        )
        self.weight = nn.Parameter(torch.empty(d_ff * d_model - 1))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_index],
                self.weight.new_zeros(1),
                self.weight[self.missing_index :],
            )
        )
        return flat.view(self.d_ff, self.d_model)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        return self.drop(self.fc2(F.gelu(hidden)))
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = GaugeFixedMLPLinear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.full_weight(), bias)
        return self.drop(self.fc2(F.gelu(hidden)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedKeyLinear):
            d_model = module.d_model
=======
        elif isinstance(module, GaugeFixedMLPLinear):
            full = module.weight.new_empty(module.d_ff, module.d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                row = full[module.gauge_row]
                omitted = row[-1].clone()
                row[:-1].sub_(omitted)
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
        elif isinstance(module, GaugeFixedKeyLinear):
            d_model = module.d_model
>>>>>>> REPLACE

<<<<<<< SEARCH
        return loss


@torch.no_grad()
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
    key_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Parameter, int]
    ],
    max_norm: float,
) -> None:
=======
        return loss


def full_mlp_gradient(
    module: torch.nn.Module,
    ln_scale: torch.nn.Parameter,
    ln_bias: torch.nn.Parameter,
) -> torch.Tensor:
    parameter = module.weight
    full_grad = torch.cat(
        (
            parameter.grad[: module.missing_index],
            parameter.grad.new_zeros(1),
            parameter.grad[module.missing_index :],
        )
    ).view(module.d_ff, module.d_model)

    gamma = ln_scale.detach()
    shift = (ln_bias.detach() / gamma).sum()
    row = module.gauge_row
    full_grad[row, -1] = gamma[-1] * (
        module.bias.grad[row] * shift
        - (full_grad[row, :-1] / gamma[:-1]).sum()
    )
    return full_grad


class MLPGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for one LayerNorm-affine MLP quotient."""

    def __init__(
        self,
        module: torch.nn.Module,
        ln_scale: torch.nn.Parameter,
        ln_bias: torch.nn.Parameter,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.module = module
        self.ln_scale = ln_scale
        self.ln_bias = ln_bias
        super().__init__(
            [module.weight, module.bias],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        weight = self.module.weight
        bias = self.module.bias
        if weight.grad is None or bias.grad is None:
            return loss

        full_grad = full_mlp_gradient(
            self.module, self.ln_scale, self.ln_bias
        )
        weight_state = self.state[weight]
        if not weight_state:
            weight_state["step"] = 0
            weight_state["exp_avg"] = torch.zeros_like(full_grad)
            weight_state["exp_avg_sq"] = torch.zeros_like(full_grad)

        weight_state["step"] += 1
        weight_avg = weight_state["exp_avg"]
        weight_avg_sq = weight_state["exp_avg_sq"]
        weight_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
        weight_avg_sq.mul_(beta2).addcmul_(
            full_grad, full_grad, value=1.0 - beta2
        )

        weight_step = weight_state["step"]
        weight_correction1 = 1.0 - beta1 ** weight_step
        weight_correction2 = 1.0 - beta2 ** weight_step
        weight_denom = weight_avg_sq.sqrt().div_(
            math.sqrt(weight_correction2)
        ).add_(group["eps"])

        full_value = self.module.full_weight()
        full_value.mul_(1.0 - group["lr"] * group["weight_decay"])
        full_value.addcdiv_(
            weight_avg,
            weight_denom,
            value=-group["lr"] / weight_correction1,
        )

        bias_grad = bias.grad
        bias_state = self.state[bias]
        if not bias_state:
            bias_state["step"] = 0
            bias_state["exp_avg"] = torch.zeros_like(bias)
            bias_state["exp_avg_sq"] = torch.zeros_like(bias)

        bias_state["step"] += 1
        bias_avg = bias_state["exp_avg"]
        bias_avg_sq = bias_state["exp_avg_sq"]
        bias_avg.mul_(beta1).add_(bias_grad, alpha=1.0 - beta1)
        bias_avg_sq.mul_(beta2).addcmul_(
            bias_grad, bias_grad, value=1.0 - beta2
        )

        bias_step = bias_state["step"]
        bias_correction1 = 1.0 - beta1 ** bias_step
        bias_correction2 = 1.0 - beta2 ** bias_step
        bias_denom = bias_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(group["eps"])

        bias.mul_(1.0 - group["lr"] * group["weight_decay"])
        bias.addcdiv_(
            bias_avg,
            bias_denom,
            value=-group["lr"] / bias_correction1,
        )

        gamma = self.ln_scale.detach()
        shift = (self.ln_bias.detach() / gamma).sum()
        row = full_value[self.module.gauge_row]
        omitted = row[-1].clone()
        row[:-1].sub_(omitted * gamma[-1] / gamma[:-1])
        row[-1].zero_()
        bias[self.module.gauge_row].add_(
            omitted * gamma[-1] * shift
        )

        flat = full_value.reshape(-1)
        weight.copy_(
            torch.cat(
                (
                    flat[: self.module.missing_index],
                    flat[self.module.missing_index + 1 :],
                )
            )
        )
        return loss


@torch.no_grad()
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
    key_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Parameter, int]
    ],
    mlp_gauges: List[
        Tuple[torch.nn.Module, torch.nn.Parameter, torch.nn.Parameter]
    ],
    max_norm: float,
) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
        for key_parameter, ln_scale, d_model in key_gauges:
            if parameter is key_parameter:
                full_grad = full_key_gradient(
                    key_parameter, ln_scale, d_model
                )
                total_sq.add_(
                    full_grad[d_model, -1].float().square()
                )

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
=======
        for key_parameter, ln_scale, d_model in key_gauges:
            if parameter is key_parameter:
                full_grad = full_key_gradient(
                    key_parameter, ln_scale, d_model
                )
                total_sq.add_(
                    full_grad[d_model, -1].float().square()
                )

        for module, ln_scale, ln_bias in mlp_gauges:
            if parameter is module.weight:
                full_grad = full_mlp_gradient(
                    module, ln_scale, ln_bias
                )
                total_sq.add_(
                    full_grad[
                        module.gauge_row, -1
                    ].float().square()
                )

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_gauge_modules = [
        (block.attn.qkv, block.ln1.weight)
        for block in model.blocks
    ]
    excluded = {id(position_parameter)}
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
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
    key_optimizers = [
        KeyGaugeAdamW(
            module,
            ln_scale,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for module, ln_scale in key_gauge_modules
    ]
    all_optimizers = [optimizer, position_optimizer, *key_optimizers]
    key_gauges = [
        (module.weight, ln_scale, module.d_model)
        for module, ln_scale in key_gauge_modules
    ]
=======
    key_gauge_modules = [
        (block.attn.qkv, block.ln1.weight)
        for block in model.blocks
    ]
    mlp_gauge_modules = [
        (block.mlp.fc1, block.ln2.weight, block.ln2.bias)
        for block in model.blocks
    ]
    excluded = {id(position_parameter)}
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
    excluded.update(
        id(parameter)
        for module, _, _ in mlp_gauge_modules
        for parameter in (module.weight, module.bias)
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
    key_optimizers = [
        KeyGaugeAdamW(
            module,
            ln_scale,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for module, ln_scale in key_gauge_modules
    ]
    mlp_optimizers = [
        MLPGaugeAdamW(
            module,
            ln_scale,
            ln_bias,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for module, ln_scale, ln_bias in mlp_gauge_modules
    ]
    all_optimizers = [
        optimizer,
        position_optimizer,
        *key_optimizers,
        *mlp_optimizers,
    ]
    key_gauges = [
        (module.weight, ln_scale, module.d_model)
        for module, ln_scale in key_gauge_modules
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_grad_norm_with_gauges(
                model,
                position_parameter,
                key_gauges,
                train_cfg.grad_clip,
            )
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        position_optimizer.step()
        optimizer.step()
=======
            clip_grad_norm_with_gauges(
                model,
                position_parameter,
                key_gauges,
                mlp_gauge_modules,
                train_cfg.grad_clip,
            )
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for mlp_optimizer in mlp_optimizers:
            mlp_optimizer.step()
        position_optimizer.step()
        optimizer.step()
>>>>>>> REPLACE