MECHANISM: Second attention-output common-shift quotient

HYPOTHESIS: Removing a second common-output coordinate from the attention projection will reduce the verified model from 1606 to 1605 parameters while retaining at least 99% accuracy, because each projection input column has an independent residual-stream shift symmetry and the first attention-output quotient achieved 99.98% accuracy.

INTENDED_EDIT: Gauge-fix row 4 in the final two attention-output projection columns and generalize virtual AdamW and gradient clipping to reconstruct and optimize multiple omitted common-output coordinates per projection.

EVIDENCE: The latest verified design reached 99.98% accuracy after removing one attention-output coordinate. The only prior two-coordinate common-output trial timed out rather than failing accuracy, so extending the newly successful attention-specific quotient is the most informative next reduction.

<<<<<<< SEARCH
class GaugeFixedAttentionOutput(nn.Module):
    """Attention output projection with one residual-shift gauge fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_weight_row = min(4, out_features - 1)
        self.fixed_weight_column = in_features - 1
        self.fixed_weight_index = (
            self.fixed_weight_row * in_features + self.fixed_weight_column
        )

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(in_features, out_features, bias=False)
        self.weight = nn.Parameter(self._reduce_weight(source.weight.detach()))

    def _weight_keep_mask(self, device: torch.device) -> torch.Tensor:
        keep = torch.ones(
            self.out_features * self.in_features,
            dtype=torch.bool,
            device=device,
        )
        keep[self.fixed_weight_index] = False
        return keep

    def _reduce_weight(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        anchor = gauged[
            self.fixed_weight_row,
            self.fixed_weight_column,
        ].clone()
        gauged[:, self.fixed_weight_column].sub_(anchor)
        flat = gauged.reshape(-1)
        return flat[self._weight_keep_mask(flat.device)].clone()
=======
class GaugeFixedAttentionOutput(nn.Module):
    """Attention output projection with two residual-shift gauges fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        fixed_row = min(4, out_features - 1)
        self.fixed_weight_rows = (fixed_row, fixed_row)
        self.fixed_weight_columns = (in_features - 1, in_features - 2)
        self.fixed_weight_indices = tuple(
            row * in_features + column
            for row, column in zip(
                self.fixed_weight_rows,
                self.fixed_weight_columns,
            )
        )

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(in_features, out_features, bias=False)
        self.weight = nn.Parameter(self._reduce_weight(source.weight.detach()))

    def _weight_keep_mask(self, device: torch.device) -> torch.Tensor:
        keep = torch.ones(
            self.out_features * self.in_features,
            dtype=torch.bool,
            device=device,
        )
        keep[list(self.fixed_weight_indices)] = False
        return keep

    def _reduce_weight(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        for row, column in zip(
            self.fixed_weight_rows,
            self.fixed_weight_columns,
        ):
            anchor = gauged[row, column].clone()
            gauged[:, column].sub_(anchor)
        flat = gauged.reshape(-1)
        return flat[self._weight_keep_mask(flat.device)].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPOutputWeightGaugeAdamW:
    """AdamW with a virtual coordinate for a common-output weight direction."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [parameter for parameter, _, _, _, _ in self.gauges]
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
            for parameter, _, _, _, _ in self.gauges
        }
=======
class MLPOutputWeightGaugeAdamW:
    """AdamW with virtual coordinates for common-output weight directions."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [parameter for parameter, _, _, _, _ in self.gauges]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {
            parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    parameter.numel() + len(fixed_rows),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + len(fixed_rows),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter, _, _, fixed_rows, _ in self.gauges
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
    @torch.no_grad()
    def step(self) -> None:
        for (
            parameter,
            out_features,
            in_features,
            fixed_row,
            fixed_column,
        ) in self.gauges:
            if parameter.grad is None:
                continue

            fixed_index = fixed_row * in_features + fixed_column
            full_numel = parameter.numel() + 1
            keep = torch.ones(
                full_numel,
                dtype=torch.bool,
                device=parameter.device,
            )
            keep[fixed_index] = False

            virtual_grad = parameter.grad.new_zeros(full_numel)
            virtual_grad[keep] = parameter.grad.detach().reshape(-1)
            virtual_matrix = virtual_grad.view(out_features, in_features)
            virtual_grad[fixed_index] = -virtual_matrix[
                :, fixed_column
            ].sum()

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
            direction = exp_avg / (
                exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )

            quotient_full = direction.clone()
            quotient_full.view(
                out_features,
                in_features,
            )[:, fixed_column].sub_(direction[fixed_index])

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                quotient_full[keep].view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )
=======
    @torch.no_grad()
    def step(self) -> None:
        for (
            parameter,
            out_features,
            in_features,
            fixed_rows,
            fixed_columns,
        ) in self.gauges:
            if parameter.grad is None:
                continue

            fixed_indices = tuple(
                row * in_features + column
                for row, column in zip(fixed_rows, fixed_columns)
            )
            full_numel = parameter.numel() + len(fixed_indices)
            keep = torch.ones(
                full_numel,
                dtype=torch.bool,
                device=parameter.device,
            )
            keep[list(fixed_indices)] = False

            virtual_grad = parameter.grad.new_zeros(full_numel)
            virtual_grad[keep] = parameter.grad.detach().reshape(-1)
            virtual_matrix = virtual_grad.view(out_features, in_features)
            for fixed_index, fixed_column in zip(
                fixed_indices,
                fixed_columns,
            ):
                virtual_grad[fixed_index] = -virtual_matrix[
                    :, fixed_column
                ].sum()

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
            direction = exp_avg / (
                exp_avg_sq.sqrt().div(
                    math.sqrt(bias_correction2)
                ).add(self.eps)
            )

            quotient_full = direction.clone()
            quotient_matrix = quotient_full.view(
                out_features,
                in_features,
            )
            for fixed_index, fixed_column in zip(
                fixed_indices,
                fixed_columns,
            ):
                quotient_matrix[:, fixed_column].sub_(
                    direction[fixed_index]
                )

            parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.add_(
                quotient_full[keep].view_as(parameter),
                alpha=-self.lr / bias_correction1,
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Recover omitted MLP output-weight gradients from common-shift invariance.
    for (
        parameter,
        out_features,
        in_features,
        fixed_row,
        fixed_column,
    ) in output_weight_gauges:
        if parameter.grad is not None:
            fixed_index = fixed_row * in_features + fixed_column
            virtual_grad = parameter.grad.detach().float().new_zeros(
                parameter.numel() + 1
            )
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=virtual_grad.device,
            )
            keep[fixed_index] = False
            virtual_grad[keep] = parameter.grad.detach().reshape(-1).float()
            omitted = -virtual_grad.view(
                out_features,
                in_features,
            )[:, fixed_column].sum()
            total_sq.add_(omitted.pow(2))
=======
    # Recover omitted output-weight gradients from common-shift invariance.
    for (
        parameter,
        out_features,
        in_features,
        fixed_rows,
        fixed_columns,
    ) in output_weight_gauges:
        if parameter.grad is not None:
            fixed_indices = tuple(
                row * in_features + column
                for row, column in zip(fixed_rows, fixed_columns)
            )
            virtual_grad = parameter.grad.detach().float().new_zeros(
                parameter.numel() + len(fixed_indices)
            )
            keep = torch.ones(
                virtual_grad.numel(),
                dtype=torch.bool,
                device=virtual_grad.device,
            )
            keep[list(fixed_indices)] = False
            virtual_grad[keep] = parameter.grad.detach().reshape(-1).float()
            virtual_matrix = virtual_grad.view(out_features, in_features)
            for fixed_column in fixed_columns:
                omitted = -virtual_matrix[:, fixed_column].sum()
                total_sq.add_(omitted.pow(2))
>>>>>>> REPLACE

<<<<<<< SEARCH
    output_weight_gauges = [
        (
            block.attn.proj.weight,
            block.attn.proj.out_features,
            block.attn.proj.in_features,
            block.attn.proj.fixed_weight_row,
            block.attn.proj.fixed_weight_column,
        )
        for block in model.blocks
    ] + [
        (
            block.mlp.fc2.weight,
            block.mlp.fc2.out_features,
            block.mlp.fc2.in_features,
            block.mlp.fc2.fixed_weight_row,
            block.mlp.fc2.fixed_weight_column,
        )
        for block in model.blocks
    ]
=======
    output_weight_gauges = [
        (
            block.attn.proj.weight,
            block.attn.proj.out_features,
            block.attn.proj.in_features,
            block.attn.proj.fixed_weight_rows,
            block.attn.proj.fixed_weight_columns,
        )
        for block in model.blocks
    ] + [
        (
            block.mlp.fc2.weight,
            block.mlp.fc2.out_features,
            block.mlp.fc2.in_features,
            (block.mlp.fc2.fixed_weight_row,),
            (block.mlp.fc2.fixed_weight_column,),
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE