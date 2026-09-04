MECHANISM: Optimizer-faithful projected fourth-column gauge

HYPOTHESIS: Constraining the fourth `fc2` column while updating it through full-coordinate AdamW moments will produce a 1,579-parameter model with at least 99% accuracy by preserving the verified 1,580-parameter model’s functional optimization trajectory modulo the LayerNorm-invisible common mode.

INTENDED_EDIT: Store the fourth `fc2` column in seven zero-mean coordinates and train those coordinates with a projected AdamW optimizer that maintains the eight coordinatewise moments used by the successful baseline.

EVIDENCE: The RNG-aligned fourth-column constraint fell to 74.27% even though the same exact gauge succeeded for three columns at 99.89%, indicating that AdamW’s coordinate-dependent optimization geometry—not representational capacity or initialization—is the most relevant remaining difference.

<<<<<<< SEARCH
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and three zero-mean weight columns."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        basis = self.weight.detach().new_zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("weight_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :3]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 3:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 3 * (self.out_features - 1)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 3
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
=======
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and four zero-mean weight columns."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        basis = self.weight.detach().new_zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("weight_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :3]
        fourth_weight_coords = basis.transpose(0, 1) @ full_weight[:, 3:4]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 4:].flatten()))
        )
        self.fourth_weight = nn.Parameter(fourth_weight_coords.flatten())
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 3 * (self.out_features - 1)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        fourth_weight = self.weight_basis @ self.fourth_weight.view(
            self.out_features - 1, 1
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 4
        )
        weight = torch.cat(
            (leading_weight, fourth_weight, remaining_weight), dim=1
        )
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, OutputAnchoredLinear):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 1)
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            baseline_leading = baseline_weight[
                : 2 * (module.out_features - 1)
            ].view(module.out_features - 1, 2)
            baseline_remaining = baseline_weight[
                2 * (module.out_features - 1) :
            ].view(module.out_features, module.in_features - 2)
            third_weight_coords = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, :1]
            )
            compact_weight = torch.cat(
                (
                    torch.cat(
                        (baseline_leading, third_weight_coords), dim=1
                    ).flatten(),
                    baseline_remaining[:, 1:].flatten(),
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
=======
        if isinstance(module, OutputAnchoredLinear):
            baseline_weight = module.weight.new_empty(
                2 * (module.out_features - 1)
                + module.out_features * (module.in_features - 2)
            )
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            baseline_leading = baseline_weight[
                : 2 * (module.out_features - 1)
            ].view(module.out_features - 1, 2)
            baseline_remaining = baseline_weight[
                2 * (module.out_features - 1) :
            ].view(module.out_features, module.in_features - 2)
            third_weight_coords = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, :1]
            )
            fourth_weight_coords = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, 1:2]
            )
            compact_weight = torch.cat(
                (
                    torch.cat(
                        (baseline_leading, third_weight_coords), dim=1
                    ).flatten(),
                    baseline_remaining[:, 2:].flatten(),
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.fourth_weight.copy_(fourth_weight_coords.flatten())
                module.bias.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
from src.model import ModelConfig, TinyDecoderLM, count_parameters
=======
from src.model import ModelConfig, OutputAnchoredLinear, TinyDecoderLM, count_parameters
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
class ProjectedAdamW(torch.optim.Optimizer):
    """AdamW in full coordinates, projected onto a zero-mean model parameter."""

    def __init__(
        self,
        parameter_bases,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        pairs = list(parameter_bases)
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__([parameter for parameter, _ in pairs], defaults)
        self.basis_by_id = {
            id(parameter): basis for parameter, basis in pairs
        }

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
                if parameter.grad.is_sparse:
                    raise RuntimeError("ProjectedAdamW does not support sparse gradients")

                basis = self.basis_by_id[id(parameter)]
                full_grad = basis @ parameter.grad
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

                full_weight = basis @ parameter
                full_weight.mul_(1.0 - group["lr"] * group["weight_decay"])
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (
                    exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                ).add_(group["eps"])
                full_weight.addcdiv_(
                    exp_avg,
                    denom,
                    value=-group["lr"] / bias_correction1,
                )
                parameter.copy_(basis.transpose(0, 1) @ full_weight)

        return loss


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    projected_pairs = [
        (module.fourth_weight, module.weight_basis)
        for module in model.modules()
        if isinstance(module, OutputAnchoredLinear)
    ]
    projected_ids = {id(parameter) for parameter, _ in projected_pairs}
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in projected_ids
    ]
    optimizer = torch.optim.AdamW(
        regular_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    projected_optimizer = ProjectedAdamW(
        projected_pairs,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        for pg in projected_optimizer.param_groups:
            pg["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        projected_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        projected_optimizer.step()
>>>>>>> REPLACE