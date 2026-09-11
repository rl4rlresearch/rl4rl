MECHANISM: Dense-coordinate AdamW on a zero-mean final-MLP bias quotient

HYPOTHESIS: Projecting ordinary eight-coordinate AdamW updates onto an orthonormal seven-parameter final-MLP bias will produce a 1,572-parameter model with at least 99% accuracy.

INTENDED_EDIT: Apply the successful orthonormal common-bias gauge to the attention projection and final MLP output, while training the latter with full-coordinate AdamW moments projected into its seven-dimensional learned parameterization.

EVIDENCE: The orthonormal attention-projection gauge reached 99.97% at 1,573 parameters; the orthonormal final-MLP gauge narrowly missed at 98.89%, suggesting its exact null-direction removal preserves capacity but its rotated elementwise-Adam geometry needs correction.

<<<<<<< SEARCH
        return F.linear(x, weight, fused_bias)


class CausalSelfAttention(nn.Module):
=======
        return F.linear(x, weight, fused_bias)


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


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = OrthonormalCommonBiasGaugedLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = OrthonormalCommonBiasGaugedLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, LayerNormGaugedQKV):
            with torch.no_grad():
=======
        elif isinstance(module, OrthonormalCommonBiasGaugedLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNormGaugedQKV):
            with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
from src.model import ModelConfig, TinyDecoderLM, count_parameters
=======
from src.model import ModelConfig, TinyDecoderLM, count_parameters
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
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


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

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

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        step_dense_bias_quotient(
            quotient_bias_modules,
            quotient_bias_states,
            lr_now,
            train_cfg.weight_decay,
        )
>>>>>>> REPLACE