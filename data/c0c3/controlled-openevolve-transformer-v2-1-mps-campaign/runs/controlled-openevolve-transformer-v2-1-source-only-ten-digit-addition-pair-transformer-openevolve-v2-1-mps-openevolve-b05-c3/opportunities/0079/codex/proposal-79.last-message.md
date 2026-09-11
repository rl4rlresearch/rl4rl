MECHANISM: Pre-MLP LayerNorm-nullspace weight quotient

HYPOTHESIS: Gauge-fixing one input coefficient in every `fc1` row will reduce the verified 1,298-parameter design to 1,286 parameters while retaining at least 99% accuracy, because bias-free LayerNorm outputs have an exact one-dimensional nullspace that already supported the verified value-map quotient.

INTENDED_EDIT: Adopt the verified seven-scale `ln1`, value-map quotient, and `ln2`-bias absorption, then represent each `fc1` row with seven coefficients and train both input-map quotients using full-space AdamW moments, weight decay, and gauge-aware clipping.

EVIDENCE: The 1,298-parameter reference achieved 99.92% accuracy, and its analogous eight-parameter value-map nullspace quotient achieved 99.89% in isolation; the earlier combined 1,286-parameter attempt could not be verified, so applying only the remaining `fc1` quotient atop the qualified staged base is the most informative next test.

<<<<<<< SEARCH
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with six scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 6))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(6)))
=======
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with seven scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(7)))
>>>>>>> REPLACE

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


class GaugeFixedFC1Linear(nn.Linear):
    """MLP input map modulo its LayerNorm-nullspace row shifts."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_model, d_ff)
        self.d_model = d_model
        self.d_ff = d_ff
        self.weight = nn.Parameter(torch.empty(d_ff, d_model - 1))

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (self.weight, self.weight.new_zeros(self.d_ff, 1)),
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight(), self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.value = FixedRouteValueLinear(d_model)
=======
        self.value = GaugeFixedValueLinear(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = GaugeFixedFC1Linear(d_model, d_ff)
        self.fc2 = GaugeFixedMLPProjectionLinear(d_model, d_ff)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        output_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
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
        elif isinstance(module, GaugeFixedFC1Linear):
            full = module.weight.new_empty(
                module.d_ff, module.d_model
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                module.weight.copy_(full[:, :-1])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    last_coordinate_gauges: List[torch.nn.Parameter],
    key_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Parameter, int]
    ],
    projection_gauges: List[Tuple[torch.nn.Parameter, int]],
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        for projection_parameter, d_model in projection_gauges:
            if parameter is projection_parameter:
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_gauge_modules = []
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
    excluded.update(id(module.weight) for module in projection_gauge_modules)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    all_optimizers = [
        optimizer,
        *last_coordinate_optimizers,
        *key_optimizers,
        *projection_optimizers,
=======
    all_optimizers = [
        optimizer,
        *last_coordinate_optimizers,
        *key_optimizers,
        *input_optimizers,
        *projection_optimizers,
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_gauges = [
        (module.weight, ln_scale, module.d_model)
        for module, ln_scale in key_gauge_modules
    ]
    projection_gauges = [
=======
    key_gauges = [
        (module.weight, ln_scale, module.d_model)
        for module, ln_scale in key_gauge_modules
    ]
    input_gauges = [
        (module.weight, ln_module)
        for module, ln_module in input_gauge_modules
    ]
    projection_gauges = [
>>>>>>> REPLACE

<<<<<<< SEARCH
                last_coordinate_gauge_parameters,
                key_gauges,
                projection_gauges,
=======
                last_coordinate_gauge_parameters,
                key_gauges,
                input_gauges,
                projection_gauges,
>>>>>>> REPLACE

<<<<<<< SEARCH
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for projection_optimizer in projection_optimizers:
=======
        for key_optimizer in key_optimizers:
            key_optimizer.step()
        for input_optimizer in input_optimizers:
            input_optimizer.step()
        for projection_optimizer in projection_optimizers:
>>>>>>> REPLACE