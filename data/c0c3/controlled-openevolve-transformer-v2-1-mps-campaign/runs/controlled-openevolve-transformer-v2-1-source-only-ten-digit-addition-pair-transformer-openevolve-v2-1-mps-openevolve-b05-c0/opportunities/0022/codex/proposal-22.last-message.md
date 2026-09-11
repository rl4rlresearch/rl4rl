MECHANISM: Cross-head quotient-aware key-projection gauge fixing

HYPOTHESIS: Removing the softmax-invisible coordinate from the first key row of the second attention head will reduce the model from 1632 to 1631 parameters while retaining at least 99% accuracy, because it distributes the two exact key gauges across heads while preserving both omitted coordinates’ full-space AdamW dynamics.

INTENDED_EDIT: Extend `GaugeFixedQKV` to omit the final input coordinate from the first key row of each head, and generalize virtual-gradient reconstruction, AdamW updates, and clipping to both coordinates.

EVIDENCE: The first quotient-aware key-coordinate removal achieved 99.92% at 1632 parameters, whereas removing the adjacent second key row reached 87.60%; with four coordinates per head, that adjacent row shares the first head, motivating the smallest alternative titration in the other head.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """QKV projection with one softmax-invisible key coordinate removed."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.fixed_index = d_model * d_model + d_model - 1

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(d_model, 3 * d_model, bias=False)
        full_weight = source.weight.detach().clone()
        self.weight = nn.Parameter(self._reduce(full_weight))

    def _reduce(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        anchor = gauged[self.d_model, -1].clone()
        gauged[self.d_model].sub_(anchor)
        flat = gauged.reshape(-1)
        return torch.cat((flat[: self.fixed_index], flat[self.fixed_index + 1 :]))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.fixed_index],
                self.weight.new_zeros(1),
                self.weight[self.fixed_index :],
            )
        )
        return flat.view(3 * self.d_model, self.d_model)

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        self.weight.copy_(self._reduce(full_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())
=======
class GaugeFixedQKV(nn.Module):
    """QKV projection with one softmax-invisible coordinate per head removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (d_model, d_model + second_offset)
        self.fixed_indices = tuple(
            row * d_model + d_model - 1 for row in self.fixed_rows
        )

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(d_model, 3 * d_model, bias=False)
        full_weight = source.weight.detach().clone()
        self.weight = nn.Parameter(self._reduce(full_weight))

    def _keep_mask(self, device: torch.device) -> torch.Tensor:
        keep = torch.ones(
            3 * self.d_model * self.d_model,
            dtype=torch.bool,
            device=device,
        )
        keep[list(self.fixed_indices)] = False
        return keep

    def _reduce(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        for row in self.fixed_rows:
            anchor = gauged[row, -1].clone()
            gauged[row].sub_(anchor)
        flat = gauged.reshape(-1)
        return flat[self._keep_mask(flat.device)].clone()

    def full_weight(self) -> torch.Tensor:
        keep = self._keep_mask(self.weight.device)
        flat = self.weight.new_zeros(keep.numel())
        flat = flat.masked_scatter(keep, self.weight)
        return flat.view(3 * self.d_model, self.d_model)

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        self.weight.copy_(self._reduce(full_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model)
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.parameters = [parameter for parameter, _, _ in self.gauges]
=======
        self.parameters = [parameter for parameter, _, _, _ in self.gauges]
>>>>>>> REPLACE

<<<<<<< SEARCH
                "exp_avg": torch.zeros(
                    parameter.numel() + 1,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + 1,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
=======
                "exp_avg": torch.zeros(
                    parameter.numel() + 2,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + 2,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
>>>>>>> REPLACE

<<<<<<< SEARCH
    @torch.no_grad()
    def step(self) -> None:
        for parameter, ln_scale, d_model in self.gauges:
            if parameter.grad is None:
                continue

            fixed_index = d_model * d_model + d_model - 1
            row_start = d_model * d_model
            grad = parameter.grad.detach().reshape(-1)
            virtual_grad = torch.cat(
                (
                    grad[:fixed_index],
                    grad.new_zeros(1),
                    grad[fixed_index:],
                )
            )

            scale = ln_scale.detach().reshape(-1)
            virtual_grad[fixed_index] = -scale[-1] * (
                virtual_grad[row_start:fixed_index] / scale[:-1]
            ).sum()

            state = self.state[parameter]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                virtual_grad,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad,
                virtual_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(self.eps)
            direction = exp_avg / denom

            quotient_direction = torch.cat(
                (direction[:fixed_index], direction[fixed_index + 1 :])
            ).clone()
            quotient_direction[row_start:fixed_index].sub_(
                direction[fixed_index] * scale[-1] / scale[:-1]
            )

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                quotient_direction.view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )
=======
    @torch.no_grad()
    def step(self) -> None:
        for parameter, ln_scale, d_model, fixed_rows in self.gauges:
            if parameter.grad is None:
                continue

            fixed_indices = tuple(
                row * d_model + d_model - 1 for row in fixed_rows
            )
            full_numel = parameter.numel() + len(fixed_indices)
            keep = torch.ones(
                full_numel,
                dtype=torch.bool,
                device=parameter.device,
            )
            keep[list(fixed_indices)] = False

            grad = parameter.grad.detach().reshape(-1)
            virtual_grad = grad.new_zeros(full_numel)
            virtual_grad[keep] = grad

            scale = ln_scale.detach().reshape(-1)
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                virtual_grad[fixed_index] = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()

            state = self.state[parameter]
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(self.beta1).add_(
                virtual_grad,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                virtual_grad,
                virtual_grad,
                value=1.0 - self.beta2,
            )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(self.eps)
            direction = exp_avg / denom

            quotient_full = direction.clone()
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                quotient_full[row_start:fixed_index].sub_(
                    direction[fixed_index] * scale[-1] / scale[:-1]
                )
            quotient_direction = quotient_full[keep]

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                quotient_direction.view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Recover the omitted key gradient from the LayerNorm-null direction.
    for parameter, ln_scale, d_model in key_gauges:
        if parameter.grad is not None:
            fixed_index = d_model * d_model + d_model - 1
            row_start = d_model * d_model
            grad = parameter.grad.detach().reshape(-1).float()
            scale = ln_scale.detach().reshape(-1).float()
            omitted = -scale[-1] * (
                grad[row_start:fixed_index] / scale[:-1]
            ).sum()
            total_sq.add_(omitted.pow(2))
=======
    # Recover the omitted key gradients from the LayerNorm-null directions.
    for parameter, ln_scale, d_model, fixed_rows in key_gauges:
        if parameter.grad is not None:
            fixed_indices = tuple(
                row * d_model + d_model - 1 for row in fixed_rows
            )
            grad = parameter.grad.detach().reshape(-1).float()
            virtual_grad = grad.new_zeros(grad.numel() + len(fixed_indices))
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=grad.device,
            )
            keep[list(fixed_indices)] = False
            virtual_grad[keep] = grad

            scale = ln_scale.detach().reshape(-1).float()
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                omitted = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()
                total_sq.add_(omitted.pow(2))
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_gauges = [
        (block.attn.qkv.weight, block.ln1.weight, model_cfg.d_model)
        for block in model.blocks
    ]
=======
    key_gauges = [
        (
            block.attn.qkv.weight,
            block.ln1.weight,
            model_cfg.d_model,
            block.attn.qkv.fixed_rows,
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            *[parameter for parameter, _, _ in key_gauges],
=======
            *[parameter for parameter, _, _, _ in key_gauges],
>>>>>>> REPLACE