MECHANISM: Second common-output MLP projection quotient on an alternate hidden feature

HYPOTHESIS: Gauge-fixing `fc2.weight[4, 0]` alongside the proven last-column quotient will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove row 4 from MLP output columns 11 and 0, then generalize reconstruction, virtual AdamW moments, quotient updates, and gradient clipping to both omitted coordinates.

EVIDENCE: One common-output MLP quotient is present in the 99.97%-accurate 1607-parameter design. The prior second quotient on the adjacent final column timed out rather than failing accuracy, motivating an independent quotient on the distant hidden column 0.

<<<<<<< SEARCH
        self.fixed_weight_row = 4
        self.fixed_weight_column = in_features - 1
        self.fixed_weight_index = (
            self.fixed_weight_row * in_features + self.fixed_weight_column
        )
=======
        self.fixed_weight_rows = (4, 4)
        self.fixed_weight_columns = (in_features - 1, 0)
        self.fixed_weight_indices = tuple(
            row * in_features + column
            for row, column in zip(
                self.fixed_weight_rows,
                self.fixed_weight_columns,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        keep[self.fixed_weight_index] = False
=======
        keep[list(self.fixed_weight_indices)] = False
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauged = full_weight.clone()
        anchor = gauged[
            self.fixed_weight_row,
            self.fixed_weight_column,
        ].clone()
        gauged[:, self.fixed_weight_column].sub_(anchor)
        flat = gauged.reshape(-1)
=======
        gauged = full_weight.clone()
        for row, column in zip(
            self.fixed_weight_rows,
            self.fixed_weight_columns,
        ):
            anchor = gauged[row, column].clone()
            gauged[:, column].sub_(anchor)
        flat = gauged.reshape(-1)
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
            }
            for parameter, _, _, _, _ in self.gauges
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            block.mlp.fc2.fixed_weight_row,
            block.mlp.fc2.fixed_weight_column,
=======
            block.mlp.fc2.fixed_weight_rows,
            block.mlp.fc2.fixed_weight_columns,
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            virtual_matrix = virtual_grad.view(
                out_features,
                in_features,
            )
            for fixed_column in fixed_columns:
                omitted = -virtual_matrix[:, fixed_column].sum()
                total_sq.add_(omitted.pow(2))
>>>>>>> REPLACE