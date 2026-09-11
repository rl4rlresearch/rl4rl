MECHANISM: Four-coordinate pre-MLP LayerNorm-scale quotient

HYPOTHESIS: Extending the verified three-coordinate `ln2` scale quotient to four coordinates will reduce the model from 1,283 to 1,282 learned parameters while retaining at least 99% accuracy, because `fc1` can absorb the additional fixed channel scale.

INTENDED_EDIT: Adopt the qualified seven-scale `ln1`, value-map and `fc1` LayerNorm-nullspace quotients, and `ln2` bias absorption, then represent `ln2` with four learned scales and four fixed unit scales while preserving full-space gauge-aware optimization.

EVIDENCE: Reference Design 3 achieved 99.98% accuracy with 1,283 parameters after the one-, two-, and three-coordinate `ln2` quotients all exceeded 99%; the next single-coordinate extension is the most direct boundary test.

<<<<<<< SEARCH
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with five scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 5))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )
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


class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with four scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 4))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )
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
        output = F.linear(
            F.gelu(hidden), self.fc2.full_weight(), output_bias
        )
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = GaugeFixedMLPScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, FixedRouteValueLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[2 * d_model :])
        elif isinstance(module, GaugeTiedHead):
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
        elif isinstance(module, GaugeTiedHead):
>>>>>>> REPLACE

<<<<<<< SEARCH
                parameter[-1].copy_(query_value)

        return loss


def full_projection_gradient(
=======
                parameter[-1].copy_(query_value)

        return loss


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
    input_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Module]
    ],
    projection_gauges: List[Tuple[torch.nn.Parameter, int]],
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

        for projection_parameter, d_model in projection_gauges:
=======
        for key_parameter, ln_scale, d_model in key_gauges:
            if parameter is key_parameter:
                full_grad = full_key_gradient(
                    key_parameter, ln_scale, d_model
                )
                total_sq.add_(
                    full_grad[d_model, -1].float().square()
                )

        for input_parameter, ln_module in input_gauges:
            if parameter is input_parameter:
                full_grad = full_input_gradient(
                    input_parameter, ln_module
                )
                total_sq.add_(
                    full_grad[:, -1].float().square().sum()
                )

        for projection_parameter, d_model in projection_gauges:
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_gauge_modules = []
    projection_gauge_modules = [
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
    excluded.update(id(module.weight) for module in projection_gauge_modules)
=======
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
        *last_coordinate_optimizers,
        *key_optimizers,
        *projection_optimizers,
=======
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