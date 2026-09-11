MECHANISM: Single attention-output weight-column shift quotient

HYPOTHESIS: Extending the qualified projection-bias shift gauge to one attention projection weight column will produce 1,595 learned parameters and retain at least 99% accuracy because the omitted coefficient changes the residual stream only by a channel-common, LayerNorm-invisible shift.

INTENDED_EDIT: Adopt the qualified three-active-query construction, then gauge-fix one attention projection weight coefficient while preserving full-shape initialization, AdamW moments, weight decay, and gradient clipping.

EVIDENCE: Reference Design 3 achieved 99.93% accuracy at 1,596 parameters by exploiting the attention projection’s common-output-shift symmetry; applying the same already-qualified symmetry to one input-dependent projection column is a conservative one-parameter extension.

<<<<<<< SEARCH
        return flat.view(3 * self.d_model, self.d_model)


class CausalSelfAttention(nn.Module):
=======
        return flat.view(3 * self.d_model, self.d_model)


class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection with one common-output-shift coordinate fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.missing_index = (d_model - 1) * d_model
        self.weight = nn.Parameter(torch.empty(d_model * d_model - 1))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_index],
                self.weight.new_zeros(1),
                self.weight[self.missing_index :],
            )
        )
        return flat.view(self.d_model, self.d_model)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. Key/value biases are omitted, and four query biases remain.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.proj = nn.Linear(d_model, d_model)
=======
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. One query coordinate is explicit and a second uses the
        # functionally redundant common shift of the projection bias.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 7))
        self.proj = GaugeFixedProjectionLinear(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
        qkv = F.linear(x, self.qkv.full_weight(), bias)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
=======
        y = F.linear(y, self.proj.full_weight(), self.proj.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, 0].clone()
                full[:, 0].sub_(omitted)
                full[-1, 0].zero_()
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
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
        return loss


def full_key_gradient(
=======
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


def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    missing_index = (d_model - 1) * d_model
    full_grad = torch.cat(
        (
            parameter.grad[:missing_index],
            parameter.grad.new_zeros(1),
            parameter.grad[missing_index:],
        )
    ).view(d_model, d_model)
    full_grad[-1, 0] = -full_grad[:-1, 0].sum()
    return full_grad


class ProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for one attention-output shift quotient."""

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
        missing_index = self.module.missing_index
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

                column = full_value[:, 0]
                omitted = column[-1].clone()
                column.sub_(omitted)
                column[-1].zero_()

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


def full_key_gradient(
>>>>>>> REPLACE

<<<<<<< SEARCH
@torch.no_grad()
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
    key_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Parameter, int]
    ],
    max_norm: float,
) -> None:
    total_sq = torch.zeros(
        (), device=position_parameter.device, dtype=torch.float32
    )
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total_sq.add_(grad.square().sum())

        if parameter is position_parameter:
            total_sq.add_(grad.sum(dim=-1).square().sum())

        for key_parameter, ln_scale, d_model in key_gauges:
            if parameter is key_parameter:
                full_grad = full_key_gradient(
                    key_parameter, ln_scale, d_model
                )
                total_sq.add_(
                    full_grad[d_model, -1].float().square()
                )

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(
                coefficient.to(dtype=parameter.grad.dtype)
            )
=======
@torch.no_grad()
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
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

        if parameter is position_parameter:
            total_sq.add_(grad.sum(dim=-1).square().sum())

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
                    full_grad[-1, 0].float().square()
                )

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(
                coefficient.to(dtype=parameter.grad.dtype)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    position_parameter = model.pos_emb.weight
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
    position_parameter = model.pos_emb.weight
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
        position_optimizer.step()
        optimizer.step()
>>>>>>> REPLACE