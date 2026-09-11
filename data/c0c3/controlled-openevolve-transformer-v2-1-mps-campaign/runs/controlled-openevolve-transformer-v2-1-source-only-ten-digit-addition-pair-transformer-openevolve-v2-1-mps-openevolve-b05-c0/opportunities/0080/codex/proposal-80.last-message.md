MECHANISM: Dual virtual-AdamW quotient of independent MLP common-output weight directions

HYPOTHESIS: Gauge-fixing `fc2.weight[4, 10]` alongside `[4, 11]` will reduce the model from 1610 to 1609 parameters while retaining at least 99% accuracy, because each affected input column can shift all output features equally and is therefore independently invisible to the final LayerNorm.

INTENDED_EDIT: Remove a second MLP output-weight coordinate, reconstruct both fixed coordinates during inference, and generalize virtual AdamW moments and gradient clipping to both common-output directions.

EVIDENCE: Removing `fc2.weight[4, 11]` with virtual-gradient and AdamW-state preservation achieved 99.96% accuracy at 1610 parameters; applying the same verified exact gauge to the adjacent independent input column is the most direct next reduction.

<<<<<<< SEARCH
class GaugeFixedMLPOutput(nn.Module):
    """MLP output projection with common-output weight and bias gauges fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_index = 4
        self.fixed_weight_row = 4
        self.fixed_weight_column = in_features - 1
        self.fixed_weight_index = (
            self.fixed_weight_row * in_features + self.fixed_weight_column
        )
=======
class GaugeFixedMLPOutput(nn.Module):
    """MLP output projection with two weight gauges and one bias gauge fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_index = 4
        self.fixed_weight_row = 4
        self.fixed_weight_columns = (in_features - 2, in_features - 1)
        self.fixed_weight_indices = tuple(
            self.fixed_weight_row * in_features + column
            for column in self.fixed_weight_columns
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        for column in self.fixed_weight_columns:
            anchor = gauged[self.fixed_weight_row, column].clone()
            gauged[:, column].sub_(anchor)
        flat = gauged.reshape(-1)
        return flat[self._weight_keep_mask(flat.device)].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPOutputWeightGaugeAdamW:
    """AdamW with a virtual coordinate for a common-output weight direction."""
=======
class MLPOutputWeightGaugeAdamW:
    """AdamW with virtual coordinates for common-output weight directions."""
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        self.state = {
            parameter: {
                "step": 0,
                "exp_avg": torch.zeros(
                    parameter.numel() + len(fixed_columns),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                "exp_avg_sq": torch.zeros(
                    parameter.numel() + len(fixed_columns),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            }
            for parameter, _, _, _, fixed_columns in self.gauges
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        for (
            parameter,
            out_features,
            in_features,
            fixed_row,
            fixed_columns,
        ) in self.gauges:
            if parameter.grad is None:
                continue

            fixed_indices = tuple(
                fixed_row * in_features + fixed_column
                for fixed_column in fixed_columns
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            quotient_full = direction.clone()
            quotient_full.view(
                out_features,
                in_features,
            )[:, fixed_column].sub_(direction[fixed_index])

            parameter.mul_(1.0 - self.lr * self.weight_decay)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    output_weight_gauges = [
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
            block.mlp.fc2.weight,
            block.mlp.fc2.out_features,
            block.mlp.fc2.in_features,
            block.mlp.fc2.fixed_weight_row,
            block.mlp.fc2.fixed_weight_columns,
        )
        for block in model.blocks
    ]
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
    # Recover omitted MLP output-weight gradients from common-shift invariance.
    for (
        parameter,
        out_features,
        in_features,
        fixed_row,
        fixed_columns,
    ) in output_weight_gauges:
        if parameter.grad is not None:
            fixed_indices = tuple(
                fixed_row * in_features + fixed_column
                for fixed_column in fixed_columns
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
            virtual_matrix = virtual_grad.view(
                out_features,
                in_features,
            )
            for fixed_column in fixed_columns:
                omitted = -virtual_matrix[:, fixed_column].sum()
                total_sq.add_(omitted.pow(2))
>>>>>>> REPLACE