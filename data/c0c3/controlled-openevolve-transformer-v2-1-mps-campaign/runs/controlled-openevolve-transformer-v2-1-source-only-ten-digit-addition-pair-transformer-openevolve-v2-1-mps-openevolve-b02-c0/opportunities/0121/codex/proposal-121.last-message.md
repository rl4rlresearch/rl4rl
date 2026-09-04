MECHANISM: Virtual-coordinate Adam for a central MLP output gauge

HYPOTHESIS: Gauging `fc2` column 6 while updating its seven coefficients through reconstructed eight-coordinate AdamW dynamics will reduce the model to 1,574 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Represent central `fc2` column 6 in the existing zero-mean basis, exclude those seven coefficients from ordinary AdamW, and apply projected updates computed from full eight-coordinate Adam moments.

EVIDENCE: The same central-column gauge reached only 47.65% with Adam applied directly in rotated coordinates, while the ungauged 1,575-parameter design reached 99.91%; reconstructing full-coordinate Adam isolates optimizer-coordinate sensitivity without changing the represented function class.

<<<<<<< SEARCH
from src.model import ModelConfig, TinyDecoderLM, count_parameters
=======
from src.model import ModelConfig, OutputAnchoredLinear, TinyDecoderLM, count_parameters
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        final_weight_coords = basis.transpose(0, 1) @ full_weight[:, -1:]
        self.weight = nn.Parameter(
            torch.cat(
                (
                    leading_weight_coords.flatten(),
                    full_weight[:, 3:-1].flatten(),
                    final_weight_coords.flatten(),
                )
            )
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 3 * (self.out_features - 1)
        middle_end = gauge_size + self.out_features * (self.in_features - 4)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        middle_weight = self.weight[gauge_size:middle_end].view(
            self.out_features, self.in_features - 4
        )
        final_weight = self.weight_basis @ self.weight[middle_end:].view(
            self.out_features - 1, 1
        )
        weight = torch.cat((leading_weight, middle_weight, final_weight), dim=1)
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
=======
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with five zero-mean columns, one using virtual Adam."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        basis = self.weight.detach().new_zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("weight_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        self.adam_gauge_column = in_features // 2
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :3]
        final_weight_coords = basis.transpose(0, 1) @ full_weight[:, -1:]
        self.weight = nn.Parameter(
            torch.cat(
                (
                    leading_weight_coords.flatten(),
                    full_weight[:, 3 : self.adam_gauge_column].flatten(),
                    full_weight[:, self.adam_gauge_column + 1 : -1].flatten(),
                    final_weight_coords.flatten(),
                )
            )
        )
        self.adam_gauge_weight = nn.Parameter(
            basis.transpose(0, 1) @ full_weight[:, self.adam_gauge_column]
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 3 * (self.out_features - 1)
        before_end = gauge_size + self.out_features * (
            self.adam_gauge_column - 3
        )
        final_start = self.weight.numel() - (self.out_features - 1)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        before_weight = self.weight[gauge_size:before_end].view(
            self.out_features, self.adam_gauge_column - 3
        )
        adam_gauge_weight = (
            self.weight_basis @ self.adam_gauge_weight
        ).unsqueeze(1)
        after_weight = self.weight[before_end:final_start].view(
            self.out_features,
            self.in_features - self.adam_gauge_column - 2,
        )
        final_weight = self.weight_basis @ self.weight[final_start:].view(
            self.out_features - 1, 1
        )
        weight = torch.cat(
            (
                leading_weight,
                before_weight,
                adam_gauge_weight,
                after_weight,
                final_weight,
            ),
            dim=1,
        )
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, OutputAnchoredLinear):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 2)
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
            final_weight_coords = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, -1:]
            )
            compact_weight = torch.cat(
                (
                    torch.cat(
                        (baseline_leading, third_weight_coords), dim=1
                    ).flatten(),
                    baseline_remaining[:, 1:-1].flatten(),
                    final_weight_coords.flatten(),
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
            final_weight_coords = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, -1:]
            )
            adam_gauge_index = module.adam_gauge_column - 2
            compact_weight = torch.cat(
                (
                    torch.cat(
                        (baseline_leading, third_weight_coords), dim=1
                    ).flatten(),
                    baseline_remaining[:, 1:adam_gauge_index].flatten(),
                    baseline_remaining[
                        :, adam_gauge_index + 1 : -1
                    ].flatten(),
                    final_weight_coords.flatten(),
                )
            )
            adam_gauge_weight = (
                module.weight_basis.transpose(0, 1)
                @ baseline_remaining[:, adam_gauge_index]
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.adam_gauge_weight.copy_(adam_gauge_weight)
                module.bias.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    virtual_adam_modules = [
        module
        for module in model.modules()
        if isinstance(module, OutputAnchoredLinear)
    ]
    virtual_adam_ids = {
        id(module.adam_gauge_weight) for module in virtual_adam_modules
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in virtual_adam_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    virtual_adam_states = [
        {
            "step": 0,
            "exp_avg": module.adam_gauge_weight.new_zeros(module.out_features),
            "exp_avg_sq": module.adam_gauge_weight.new_zeros(module.out_features),
        }
        for module in virtual_adam_modules
    ]
    virtual_beta1 = 0.9
    virtual_beta2 = 0.999
    virtual_eps = 1e-8

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
=======
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        for module in virtual_adam_modules:
            module.adam_gauge_weight.grad = None
        loss.backward()
>>>>>>> REPLACE

<<<<<<< SEARCH
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        with torch.no_grad():
            for module, state in zip(
                virtual_adam_modules, virtual_adam_states
            ):
                parameter = module.adam_gauge_weight
                if parameter.grad is None:
                    continue
                full_grad = module.weight_basis @ parameter.grad
                state["step"] += 1
                state["exp_avg"].mul_(virtual_beta1).add_(
                    full_grad, alpha=1.0 - virtual_beta1
                )
                state["exp_avg_sq"].mul_(virtual_beta2).addcmul_(
                    full_grad,
                    full_grad,
                    value=1.0 - virtual_beta2,
                )
                bias_correction1 = 1.0 - virtual_beta1 ** state["step"]
                bias_correction2 = 1.0 - virtual_beta2 ** state["step"]
                denominator = state["exp_avg_sq"].sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(virtual_eps)
                full_direction = state["exp_avg"] / denominator
                parameter.mul_(
                    1.0 - lr_now * train_cfg.weight_decay
                )
                parameter.add_(
                    module.weight_basis.transpose(0, 1) @ full_direction,
                    alpha=-lr_now / bias_correction1,
                )

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE