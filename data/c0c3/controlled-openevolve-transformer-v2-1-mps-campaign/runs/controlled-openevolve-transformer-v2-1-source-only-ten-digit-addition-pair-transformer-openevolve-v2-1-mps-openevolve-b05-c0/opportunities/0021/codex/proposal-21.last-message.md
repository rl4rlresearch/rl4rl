MECHANISM: Second quotient-aware LayerNorm key-projection gauge

HYPOTHESIS: Removing one additional softmax-invisible key-projection coordinate will reduce the model from 1632 to 1631 parameters while retaining at least 99% accuracy, because each key output row has an independent LayerNorm-induced constant-shift gauge and virtual full-coordinate AdamW preserves its optimizer dynamics.

INTENDED_EDIT: Gauge-fix the final input coordinate of a second key-projection row, reconstruct both omitted gradients and optimizer coordinates, and include both virtual coordinates in gradient clipping.

EVIDENCE: The first quotient-aware key-coordinate removal achieved 99.92% accuracy at 1632 parameters; extending that successful mechanism by exactly one coordinate is the smallest informative titration.

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
    """QKV projection with two softmax-invisible key coordinates removed."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.n_fixed = 2
        self.fixed_indices = [
            d_model * d_model + (row + 1) * d_model - 1
            for row in range(self.n_fixed)
        ]

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(d_model, 3 * d_model, bias=False)
        full_weight = source.weight.detach().clone()
        self.weight = nn.Parameter(self._reduce(full_weight))

    def _reduce(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        for row in range(self.n_fixed):
            key_row = self.d_model + row
            anchor = gauged[key_row, -1].clone()
            gauged[key_row].sub_(anchor)

        flat = gauged.reshape(-1)
        pieces = []
        start = 0
        for fixed_index in self.fixed_indices:
            pieces.append(flat[start:fixed_index])
            start = fixed_index + 1
        pieces.append(flat[start:])
        return torch.cat(pieces)

    def full_weight(self) -> torch.Tensor:
        pieces = []
        start = 0
        for offset, fixed_index in enumerate(self.fixed_indices):
            reduced_index = fixed_index - offset
            pieces.extend(
                (
                    self.weight[start:reduced_index],
                    self.weight.new_zeros(1),
                )
            )
            start = reduced_index
        pieces.append(self.weight[start:])
        return torch.cat(pieces).view(3 * self.d_model, self.d_model)

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        self.weight.copy_(self._reduce(full_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
class KeyGaugeAdamW:
    """AdamW with a virtual coordinate for a LayerNorm-null key direction."""
=======
class KeyGaugeAdamW:
    """AdamW with virtual coordinates for LayerNorm-null key directions."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauges = list(gauges)
        self.parameters = [parameter for parameter, _, _ in self.gauges]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            parameter: {
                "step": 0,
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
            }
            for parameter in self.parameters
        }
=======
        self.gauges = list(gauges)
        self.parameters = [parameter for parameter, _, _, _ in self.gauges]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    parameter.numel() + n_fixed,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + n_fixed,
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter, _, _, n_fixed in self.gauges
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        for parameter, ln_scale, d_model, n_fixed in self.gauges:
            if parameter.grad is None:
                continue

            fixed_indices = [
                d_model * d_model + (row + 1) * d_model - 1
                for row in range(n_fixed)
            ]
            grad = parameter.grad.detach().reshape(-1)
            pieces = []
            start = 0
            for offset, fixed_index in enumerate(fixed_indices):
                reduced_index = fixed_index - offset
                pieces.extend(
                    (
                        grad[start:reduced_index],
                        grad.new_zeros(1),
                    )
                )
                start = reduced_index
            pieces.append(grad[start:])
            virtual_grad = torch.cat(pieces)

            scale = ln_scale.detach().reshape(-1)
            for fixed_index in fixed_indices:
                row_start = fixed_index - d_model + 1
                virtual_grad[fixed_index] = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()
>>>>>>> REPLACE

<<<<<<< SEARCH
            quotient_direction = torch.cat(
                (direction[:fixed_index], direction[fixed_index + 1 :])
            ).clone()
            quotient_direction[row_start:fixed_index].sub_(
                direction[fixed_index] * scale[-1] / scale[:-1]
            )
=======
            for fixed_index in fixed_indices:
                row_start = fixed_index - d_model + 1
                direction[row_start:fixed_index].sub_(
                    direction[fixed_index] * scale[-1] / scale[:-1]
                )
            keep = torch.ones(
                direction.numel(),
                dtype=torch.bool,
                device=direction.device,
            )
            keep[fixed_indices] = False
            quotient_direction = direction[keep]
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
    # Recover omitted key gradients from the LayerNorm-null directions.
    for parameter, ln_scale, d_model, n_fixed in key_gauges:
        if parameter.grad is not None:
            fixed_indices = [
                d_model * d_model + (row + 1) * d_model - 1
                for row in range(n_fixed)
            ]
            grad = parameter.grad.detach().reshape(-1).float()
            scale = ln_scale.detach().reshape(-1).float()
            start = 0
            for offset, fixed_index in enumerate(fixed_indices):
                reduced_index = fixed_index - offset
                row_start = reduced_index - d_model + 1
                omitted = -scale[-1] * (
                    grad[row_start:reduced_index] / scale[:-1]
                ).sum()
                total_sq.add_(omitted.pow(2))
                start = reduced_index
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
            block.attn.qkv.n_fixed,
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            *[parameter for parameter, _, _ in key_gauges],
=======
            *[parameter for parameter, _, _, _ in key_gauges],
>>>>>>> REPLACE