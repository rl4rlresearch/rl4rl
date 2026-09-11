MECHANISM: Ambient-coordinate AdamW quotient for the fifth attention-output column

HYPOTHESIS: Constraining the fifth attention-output column to zero mean while updating it with AdamW moments in the original eight-dimensional coordinates will produce a 1,575-parameter model with at least 99% accuracy.

INTENDED_EDIT: Store seven coordinates for the fifth projection column, reconstruct it through the existing zero-mean basis, and apply a quotient-aware optimizer update that preserves the verified model’s ambient AdamW geometry and initialization draws.

EVIDENCE: The current 1,576-parameter model reached 99.92%, while the earlier naive fifth-column gauge fell to 70.06%; because AdamW’s coordinatewise second moments are not rotation-invariant, retaining ambient-coordinate moments directly targets the optimization change introduced by that failed gauge.

<<<<<<< SEARCH
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias and first four weight columns."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :4]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 4:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 4 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 4
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 4
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, self.bias_basis @ self.bias)
=======
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean bias and first five weight columns."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :4]
        fifth_weight_coords = basis.transpose(0, 1) @ full_weight[:, 4:5]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 5:].flatten()))
        )
        self.fifth_weight = nn.Parameter(fifth_weight_coords.flatten())
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 4 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 4
        )
        fifth_weight = (
            self.bias_basis @ self.fifth_weight
        ).unsqueeze(1)
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 5
        )
        weight = torch.cat(
            (leading_weight, fifth_weight, remaining_weight), dim=1
        )
        return F.linear(x, weight, self.bias_basis @ self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, OutputAnchoredLinear):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, AttentionGaugeLinear):
            baseline_weight = module.weight.new_empty(
                module.weight.numel() + module.fifth_weight.numel() + 1
            )
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            gauge_size = 4 * (module.out_features - 1)
            baseline_leading = baseline_weight[:gauge_size]
            baseline_remaining = baseline_weight[gauge_size:].view(
                module.out_features, module.in_features - 4
            )
            fifth_weight_coords = (
                module.bias_basis.transpose(0, 1)
                @ baseline_remaining[:, :1]
            )
            compact_weight = torch.cat(
                (
                    baseline_leading,
                    baseline_remaining[:, 1:].flatten(),
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.fifth_weight.copy_(fifth_weight_coords.flatten())
                module.bias.zero_()
        elif isinstance(module, OutputAnchoredLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
from src.model import ModelConfig, TinyDecoderLM, count_parameters
=======
from src.model import AttentionGaugeLinear, ModelConfig, TinyDecoderLM, count_parameters
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
@torch.no_grad()
def ambient_gauge_adamw_step(
    param: torch.Tensor,
    basis: torch.Tensor,
    state: Dict[str, torch.Tensor],
    step: int,
    lr: float,
    weight_decay: float,
) -> None:
    if param.grad is None:
        return

    beta1, beta2 = 0.9, 0.999
    ambient_grad = basis @ param.grad
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    exp_avg.mul_(beta1).add_(ambient_grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(
        ambient_grad, ambient_grad, value=1.0 - beta2
    )

    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    ambient_update = (exp_avg / bias_correction1) / (
        (exp_avg_sq / bias_correction2).sqrt() + 1e-8
    )

    param.mul_(1.0 - lr * weight_decay)
    param.add_(
        basis.transpose(0, 1) @ ambient_update,
        alpha=-lr,
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

    ambient_gauges = [
        (module.fifth_weight, module.bias_basis)
        for module in model.modules()
        if isinstance(module, AttentionGaugeLinear)
    ]
    ambient_param_ids = {id(param) for param, _ in ambient_gauges}
    optimizer = torch.optim.AdamW(
        [
            param
            for param in model.parameters()
            if id(param) not in ambient_param_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    ambient_states = [
        {
            "exp_avg": basis.new_zeros(basis.size(0)),
            "exp_avg_sq": basis.new_zeros(basis.size(0)),
        }
        for _, basis in ambient_gauges
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
        for (gauge_param, gauge_basis), gauge_state in zip(
            ambient_gauges, ambient_states
        ):
            ambient_gauge_adamw_step(
                gauge_param,
                gauge_basis,
                gauge_state,
                step + 1,
                lr_now,
                train_cfg.weight_decay,
            )
>>>>>>> REPLACE