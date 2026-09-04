MECHANISM: Optimizer-preserving reuse of the attention projection-bias shift gauge

HYPOTHESIS: Reusing the functionally redundant common shift of the attention projection bias as the second independent query-bias coordinate will produce 1,596 parameters and retain at least 99% accuracy while preserving three active query offsets.

INTENDED_EDIT: Store one query-bias scalar explicitly and the other in the final projection-bias coordinate, then use a custom AdamW update and gauge-aware clipping to reproduce independent query-bias and full projection-bias optimization.

EVIDENCE: The current three-active-coordinate construction achieved 99.89% at 1,597 parameters, while zeroing its third coordinate collapsed accuracy to 21.05%; the failed standalone output-bias quotient motivates reusing the exact shift redundancy while explicitly preserving the original virtual AdamW dynamics.

<<<<<<< SEARCH
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. Two learned coordinates generate three active query biases;
        # key/value biases and the remaining query coordinates are fixed at zero.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 6))
        self.proj = nn.Linear(d_model, d_model)
=======
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. One query coordinate is stored explicitly and a second uses
        # the functionally redundant common shift of the projection bias.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 7))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Keep the third query coordinate active by sharing the mean of the two
        # independent coordinates, rather than fixing that coordinate at zero.
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.mean().unsqueeze(0))
        )
=======
        # The projection bias's final coordinate supplies the second independent
        # query offset; its common output shift is removed by downstream norms.
        independent_query_bias = torch.cat(
            (self.qkv.bias, self.proj.bias[-1:])
        )
        query_bias = torch.cat(
            (
                independent_query_bias,
                independent_query_bias.mean().unsqueeze(0),
            )
        )
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
    shared_query_projection_parameters = [
        block.attn.proj.bias for block in model.blocks
    ]
    excluded = {id(position_parameter)}
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
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
        *shared_optimizers,
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
                shared_query_projection_parameters,
                train_cfg.grad_clip,
            )
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for shared_optimizer in shared_optimizers:
            shared_optimizer.step()
        position_optimizer.step()
        optimizer.step()
>>>>>>> REPLACE