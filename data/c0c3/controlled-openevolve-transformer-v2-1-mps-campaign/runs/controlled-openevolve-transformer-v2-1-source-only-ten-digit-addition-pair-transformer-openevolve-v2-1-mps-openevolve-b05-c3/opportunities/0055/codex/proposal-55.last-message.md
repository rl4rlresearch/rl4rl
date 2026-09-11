MECHANISM: Fifth final-MLP output-column shift quotient

HYPOTHESIS: Extending the verified four-column `fc2` gauge to a fifth adjacent hidden column will produce 1,471 learned parameters and retain at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.

INTENDED_EDIT: Omit the final output-row coefficients of the last five `fc2` input columns while preserving full-shape initialization, virtual AdamW moments, weight decay, and gauge-aware gradient clipping.

EVIDENCE: The four-column `fc2` quotient achieved 99.92% accuracy at 1,472 parameters after the one-, two-, and three-column variants all exceeded 99%; this applies the same repeatedly verified symmetry to one additional adjacent column.

<<<<<<< SEARCH
        return flat.view(self.d_model, self.d_model)


class CausalSelfAttention(nn.Module):
=======
        return flat.view(self.d_model, self.d_model)


class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with five common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 5
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 5))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(5),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
=======
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, -5:].clone()
                full[:, -5:].sub_(omitted)
                full[-1, -5:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 5 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
    missing_start = d_model * d_ff - 5
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(5),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -5:] = -full_grad[:-1, -5:].sum(dim=0)
    return full_grad


class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for five MLP common-output shift quotients."""

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

                omitted = full_value[-1, -5:].clone()
                full_value[:, -5:].sub_(omitted)
                full_value[-1, -5:].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 5 :],
                        )
                    )
                )

        return loss


def full_key_gradient(
>>>>>>> REPLACE

<<<<<<< SEARCH
    projection_gauges: List[Tuple[torch.nn.Parameter, int]],
    shared_parameters: List[torch.nn.Parameter],
=======
    projection_gauges: List[Tuple[torch.nn.Parameter, int]],
    mlp_projection_gauges: List[
        Tuple[torch.nn.Parameter, int, int]
    ],
    shared_parameters: List[torch.nn.Parameter],
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
                    full_grad[-1, -5:].float().square().sum()
                )

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    projection_gauge_modules = [
        block.attn.proj for block in model.blocks
    ]
    shared_query_projection_parameters = [
=======
    projection_gauge_modules = [
        block.attn.proj for block in model.blocks
    ]
    mlp_projection_gauge_modules = [
        block.mlp.fc2 for block in model.blocks
    ]
    shared_query_projection_parameters = [
>>>>>>> REPLACE

<<<<<<< SEARCH
    excluded.update(id(module.weight) for module in projection_gauge_modules)
    excluded.update(
        id(parameter)
        for parameter in shared_query_projection_parameters
=======
    excluded.update(id(module.weight) for module in projection_gauge_modules)
    excluded.update(
        id(module.weight) for module in mlp_projection_gauge_modules
    )
    excluded.update(
        id(parameter)
        for parameter in shared_query_projection_parameters
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
    shared_optimizers = [
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        *key_optimizers,
        *projection_optimizers,
        *shared_optimizers,
=======
        *key_optimizers,
        *projection_optimizers,
        *mlp_projection_optimizers,
        *shared_optimizers,
>>>>>>> REPLACE

<<<<<<< SEARCH
    projection_gauges = [
        (module.weight, module.d_model)
        for module in projection_gauge_modules
    ]

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
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
                key_gauges,
                projection_gauges,
                shared_query_projection_parameters,
=======
                key_gauges,
                projection_gauges,
                mlp_projection_gauges,
                shared_query_projection_parameters,
>>>>>>> REPLACE

<<<<<<< SEARCH
        for projection_optimizer in projection_optimizers:
            projection_optimizer.step()
        for shared_optimizer in shared_optimizers:
=======
        for projection_optimizer in projection_optimizers:
            projection_optimizer.step()
        for mlp_projection_optimizer in mlp_projection_optimizers:
            mlp_projection_optimizer.step()
        for shared_optimizer in shared_optimizers:
>>>>>>> REPLACE