MECHANISM: LayerNorm-nullspace value-projection quotient

HYPOTHESIS: Gauge-fixing one input coefficient in every value-projection row will reduce Reference Design 1 from 1,313 to 1,305 parameters while retaining at least 99% accuracy, because LayerNorm outputs lie in a seven-dimensional zero-sum subspace and the omitted coefficients are exact function-space redundancies.

INTENDED_EDIT: Adopt the verified seven-coordinate `ln1` scale quotient, then remove eight redundant value-projection coefficients while preserving full-shape initialization, virtual AdamW moments, weight decay, and gauge-aware clipping.

EVIDENCE: Reference Design 1 achieved 99.96% accuracy with 1,313 parameters and one adaptive `ln1` scale; complete attention-output and MLP projection gauges also retained accuracy, supporting another exact, optimizer-aware linear-map quotient without repeating the failed removal of the final adaptive scale.

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
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with seven scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(7)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )


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
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln1.bias = None
=======
        self.ln1 = GaugeFixedScaleLayerNorm(cfg.d_model)
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
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
    excluded.update(id(module.weight) for module in projection_gauge_modules)
=======
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
    excluded.update(
        id(module.weight) for module, _ in value_gauge_modules
    )
    excluded.update(id(module.weight) for module in projection_gauge_modules)
>>>>>>> REPLACE

<<<<<<< SEARCH
    projection_optimizers = [
        ProjectionGaugeAdamW(
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        *key_optimizers,
        *projection_optimizers,
=======
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
                key_gauges,
                projection_gauges,
=======
                key_gauges,
                value_gauges,
                projection_gauges,
>>>>>>> REPLACE

<<<<<<< SEARCH
        for gauge_optimizer in last_coordinate_optimizers:
            gauge_optimizer.step()
        optimizer.step()
=======
        for gauge_optimizer in last_coordinate_optimizers:
            gauge_optimizer.step()
        optimizer.step()
        for value_optimizer in value_optimizers:
            value_optimizer.step()
>>>>>>> REPLACE