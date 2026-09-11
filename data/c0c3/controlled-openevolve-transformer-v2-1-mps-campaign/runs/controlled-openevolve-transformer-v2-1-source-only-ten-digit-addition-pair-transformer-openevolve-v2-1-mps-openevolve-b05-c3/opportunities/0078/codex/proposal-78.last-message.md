MECHANISM: LayerNorm-nullspace value quotient with staged pre-MLP bias absorption

HYPOTHESIS: Building on the verified 1,305-parameter value-projection quotient, absorbing `ln2.bias` into a restored full `fc1.bias` will produce 1,298 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Adopt the verified eight-parameter value-map quotient and its gauge-aware optimizer, then remove the eight `ln2` bias parameters while restoring the twelfth `fc1` bias parameter.

EVIDENCE: Reference Design 2 achieved 99.89% accuracy with 1,305 parameters; because the combined 1,286-parameter experiment could not be verified, isolating the exact seven-parameter bias-absorption reduction avoids simultaneously introducing the untested `fc1` weight quotient.

<<<<<<< SEARCH
class FixedRouteValueLinear(nn.Linear):
    """Value projection for content-independent learned attention."""

    def __init__(self, d_model: int):
        # Preserve the original QKV constructor's RNG consumption.
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.out_features = d_model
        self.weight = nn.Parameter(torch.empty(d_model, d_model))
        self.bias = None
=======
class GaugeFixedValueLinear(nn.Linear):
    """Value map modulo its LayerNorm-nullspace row shifts."""

    def __init__(self, d_model: int):
        # Preserve the original QKV constructor's RNG consumption.
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.out_features = d_model
        self.weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (self.weight, self.weight.new_zeros(self.d_model, 1)),
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.value = FixedRouteValueLinear(d_model)
=======
        self.value = GaugeFixedValueLinear(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        input_bias = torch.cat(
            (self.fc1.bias, self.fc1.bias.new_zeros(1))
        )
        hidden = F.linear(x, self.fc1.weight, input_bias)
=======
        hidden = self.fc1(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.bias = None
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, FixedRouteValueLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[2 * d_model :])
=======
        elif isinstance(module, GaugeFixedValueLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                value = full[2 * d_model :]
                omitted = value[:, -1:].clone()
                value[:, :-1].sub_(omitted)
                value[:, -1].zero_()
                module.weight.copy_(value[:, :-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
=======
def full_value_gradient(
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
    gamma = ln_module.full_weight().detach()
    full_grad[:, -1] = -gamma[-1] * (
        full_grad[:, :-1] / gamma[:-1]
    ).sum(dim=-1)
    return full_grad


class ValueGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for value-map LayerNorm-nullspace quotients."""

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

                full_grad = full_value_gradient(
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

                gamma = self.ln_module.full_weight().detach()
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Parameter, int]
    ],
    projection_gauges: List[Tuple[torch.nn.Parameter, int]],
=======
    key_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Parameter, int]
    ],
    value_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Module]
    ],
    projection_gauges: List[Tuple[torch.nn.Parameter, int]],
>>>>>>> REPLACE

<<<<<<< SEARCH
        for projection_parameter, d_model in projection_gauges:
            if parameter is projection_parameter:
=======
        for value_parameter, ln_module in value_gauges:
            if parameter is value_parameter:
                full_grad = full_value_gradient(
                    value_parameter, ln_module
                )
                total_sq.add_(
                    full_grad[:, -1].float().square().sum()
                )

        for projection_parameter, d_model in projection_gauges:
            if parameter is projection_parameter:
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_gauge_modules = []
    projection_gauge_modules = [
=======
    key_gauge_modules = []
    value_gauge_modules = [
        (block.attn.value, block.ln1) for block in model.blocks
    ]
    projection_gauge_modules = [
>>>>>>> REPLACE

<<<<<<< SEARCH
    excluded.update(
        id(module.weight) for module, _ in key_gauge_modules
    )
    excluded.update(
        id(module.weight) for module in projection_gauge_modules
    )
=======
    excluded.update(
        id(module.weight) for module, _ in key_gauge_modules
    )
    excluded.update(
        id(module.weight) for module, _ in value_gauge_modules
    )
    excluded.update(
        id(module.weight) for module in projection_gauge_modules
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    projection_optimizers = [
        ProjectionGaugeAdamW(
            module,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for module in projection_gauge_modules
    ]
=======
    value_optimizers = [
        ValueGaugeAdamW(
            module,
            ln_module,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for module, ln_module in value_gauge_modules
    ]
    projection_optimizers = [
        ProjectionGaugeAdamW(
            module,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for module in projection_gauge_modules
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        *last_coordinate_optimizers,
        *key_optimizers,
        *projection_optimizers,
=======
        *last_coordinate_optimizers,
        *key_optimizers,
        *value_optimizers,
        *projection_optimizers,
>>>>>>> REPLACE

<<<<<<< SEARCH
    projection_gauges = [
        (module.weight, module.d_model)
        for module in projection_gauge_modules
    ]
=======
    value_gauges = [
        (module.weight, ln_module)
        for module, ln_module in value_gauge_modules
    ]
    projection_gauges = [
        (module.weight, module.d_model)
        for module in projection_gauge_modules
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
                last_coordinate_gauge_parameters,
                key_gauges,
                projection_gauges,
=======
                last_coordinate_gauge_parameters,
                key_gauges,
                value_gauges,
                projection_gauges,
>>>>>>> REPLACE

<<<<<<< SEARCH
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for projection_optimizer in projection_optimizers:
=======
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for value_optimizer in value_optimizers:
            value_optimizer.step()
        for projection_optimizer in projection_optimizers:
>>>>>>> REPLACE