MECHANISM: Fourth first-head value-row LayerNorm gauge

HYPOTHESIS: Extending the verified 1,551-parameter design to first-head value row 19 will produce a 1,550-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified common-output affine quotients and dense-coordinate AdamW updates, then gauge the remaining first-head value row alongside rows 16–18.

EVIDENCE: Gauging value row 18 reduced the verified common-output design to 1,551 parameters with 99.78% accuracy; row 19 continues the successful adjacent first-head pattern without adding a previously fragile second-head constraint.

<<<<<<< SEARCH
            2 * d_model,
            2 * d_model + 1,
        )
=======
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class OrthonormalCommonBiasGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(
            base.weight.new_empty(out_features, in_features)
        )
        self.bias = nn.Parameter(base.bias.new_empty(out_features - 1))

        # Helmert columns form an orthonormal basis for zero-mean biases.
        basis = torch.zeros(out_features, out_features - 1)
        for column in range(out_features - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias_basis @ self.bias)
=======
class OrthonormalCommonOutputGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(
            base.weight.new_empty(out_features - 1, in_features)
        )
        self.bias = nn.Parameter(base.bias.new_empty(out_features - 1))

        # Helmert columns span the zero-mean output subspace. Components shared
        # by every output coordinate are erased by downstream LayerNorm.
        basis = torch.zeros(out_features, out_features - 1)
        for column in range(out_features - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("output_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.output_basis @ self.weight
        bias = self.output_basis @ self.bias
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = OrthonormalCommonBiasGaugedLinear(d_model, d_model)
=======
        self.proj = OrthonormalCommonOutputGaugedLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = OrthonormalCommonBiasGaugedLinear(d_ff, d_model)
=======
        self.fc2 = OrthonormalCommonOutputGaugedLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, OrthonormalCommonBiasGaugedLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
=======
        elif isinstance(module, OrthonormalCommonOutputGaugedLinear):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(
                    module.output_basis.transpose(0, 1) @ full
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
@torch.no_grad()
def step_dense_bias_quotient(
    modules, states, lr: float, weight_decay: float
) -> None:
    """Apply ordinary dense-coordinate AdamW, retaining only the quotient."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for module, state in zip(modules, states):
        parameter = module.bias
        if parameter.grad is None:
            continue

        # Recover the zero-mean dense gradient. Full-coordinate second moments
        # preserve AdamW's original coordinate geometry before projection.
        dense_grad = module.bias_basis @ parameter.grad
        state["step"] += 1
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg.mul_(beta1).add_(dense_grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            dense_grad, dense_grad, value=1.0 - beta2
        )

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        denom = exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        dense_update = exp_avg / denom

        parameter.mul_(1.0 - lr * weight_decay)
        parameter.add_(
            module.bias_basis.transpose(0, 1) @ dense_update,
            alpha=-lr / bias_correction1,
        )
=======
@torch.no_grad()
def step_dense_output_quotients(
    specifications, states, lr: float, weight_decay: float
) -> None:
    """Apply dense-coordinate AdamW and retain its zero-mean quotient."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for (module, parameter_name), state in zip(specifications, states):
        parameter = getattr(module, parameter_name)
        if parameter.grad is None:
            continue

        # Matrix multiplication handles both affine weights and bias vectors.
        dense_grad = module.output_basis @ parameter.grad
        state["step"] += 1
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg.mul_(beta1).add_(dense_grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            dense_grad, dense_grad, value=1.0 - beta2
        )

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        denom = exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        dense_update = exp_avg / denom

        parameter.mul_(1.0 - lr * weight_decay)
        parameter.add_(
            module.output_basis.transpose(0, 1) @ dense_update,
            alpha=-lr / bias_correction1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    quotient_bias_modules = [block.mlp.fc2 for block in model.blocks]
    quotient_bias_ids = {
        id(module.bias) for module in quotient_bias_modules
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quotient_bias_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    quotient_bias_states = [
        {
            "step": 0,
            "exp_avg": module.bias.new_zeros(module.out_features),
            "exp_avg_sq": module.bias.new_zeros(module.out_features),
        }
        for module in quotient_bias_modules
    ]
=======
    # Attention bias retains its successful ordinary quotient optimizer.
    # Newly gauged weights and the sensitive final-MLP bias use moments in
    # their original dense output coordinates before projection.
    quotient_specifications = []
    for block in model.blocks:
        quotient_specifications.extend(
            (
                (block.attn.proj, "weight"),
                (block.mlp.fc2, "weight"),
                (block.mlp.fc2, "bias"),
            )
        )
    quotient_parameter_ids = {
        id(getattr(module, parameter_name))
        for module, parameter_name in quotient_specifications
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quotient_parameter_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    quotient_states = []
    for module, parameter_name in quotient_specifications:
        dense_shape = (
            (module.out_features, module.in_features)
            if parameter_name == "weight"
            else (module.out_features,)
        )
        quotient_states.append(
            {
                "step": 0,
                "exp_avg": getattr(module, parameter_name).new_zeros(
                    dense_shape
                ),
                "exp_avg_sq": getattr(module, parameter_name).new_zeros(
                    dense_shape
                ),
            }
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        step_dense_bias_quotient(
            quotient_bias_modules,
            quotient_bias_states,
            lr_now,
            train_cfg.weight_decay,
        )
=======
        step_dense_output_quotients(
            quotient_specifications,
            quotient_states,
            lr_now,
            train_cfg.weight_decay,
        )
>>>>>>> REPLACE