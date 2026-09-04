MECHANISM: Alternate-anchor virtual-AdamW quotient for the final key row

HYPOTHESIS: Gauge-fixing key row `d_model + 3` at feature 4 instead of the previously failed feature 7 will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because the key-row symmetry is exact but AdamW is coordinate-dependent, and feature 4 already supports successful token-position and LayerNorm-scale gauges.

INTENDED_EDIT: Add the eighth key-row quotient using feature 4 as its anchor and generalize reconstruction, virtual gradients, AdamW projection, and clipping to support per-row anchor columns.

EVIDENCE: Seven key rows anchored at feature 7 reached 99.97%, while the eighth fell to 46.34%; feature 4 remains a particularly evidence-backed alternative because existing feature-4 token-position and LayerNorm-scale quotients both train successfully.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """QKV projection with seven softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + second_offset,
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
        self.fixed_indices = tuple(
            row * d_model + d_model - 1 for row in self.fixed_rows
        )
=======
class GaugeFixedQKV(nn.Module):
    """QKV projection with eight softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + second_offset,
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
        self.fixed_columns = tuple(
            d_model // 2 if row == d_model + 3 else d_model - 1
            for row in self.fixed_rows
        )
        self.fixed_indices = tuple(
            row * d_model + column
            for row, column in zip(self.fixed_rows, self.fixed_columns)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _reduce(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        for row in self.fixed_rows:
            anchor = gauged[row, -1].clone()
            gauged[row].sub_(anchor)
        flat = gauged.reshape(-1)
        return flat[self._keep_mask(flat.device)].clone()
=======
    def _reduce(self, full_weight: torch.Tensor) -> torch.Tensor:
        gauged = full_weight.clone()
        for row, column in zip(self.fixed_rows, self.fixed_columns):
            anchor = gauged[row, column].clone()
            gauged[row].sub_(anchor)
        flat = gauged.reshape(-1)
        return flat[self._keep_mask(flat.device)].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
class KeyGaugeAdamW:
    """AdamW with a virtual coordinate for a LayerNorm-null key direction."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
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
            for parameter, _, _, fixed_rows in self.gauges
        }
=======
class KeyGaugeAdamW:
    """AdamW with virtual coordinates for LayerNorm-null key directions."""

    def __init__(
        self,
        gauges,
        lr: float,
        weight_decay: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.gauges = list(gauges)
        self.parameters = [
            parameter for parameter, _, _, _, _ in self.gauges
        ]
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
        for parameter, ln_scale, d_model, fixed_rows in self.gauges:
            if parameter.grad is None:
                continue

            fixed_indices = tuple(
                row * d_model + d_model - 1 for row in fixed_rows
            )
=======
        for (
            parameter,
            ln_scale,
            d_model,
            fixed_rows,
            fixed_columns,
        ) in self.gauges:
            if parameter.grad is None:
                continue

            fixed_indices = tuple(
                row * d_model + column
                for row, column in zip(fixed_rows, fixed_columns)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            scale = ln_scale.detach().reshape(-1)
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                virtual_grad[fixed_index] = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()
=======
            scale = ln_scale.detach().reshape(-1)
            for row, column, fixed_index in zip(
                fixed_rows,
                fixed_columns,
                fixed_indices,
            ):
                row_start = row * d_model
                row_grad = virtual_grad[
                    row_start : row_start + d_model
                ]
                virtual_grad[fixed_index] = -scale[column] * (
                    row_grad / scale
                ).sum()
>>>>>>> REPLACE

<<<<<<< SEARCH
            quotient_full = direction.clone()
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                quotient_full[row_start:fixed_index].sub_(
                    direction[fixed_index] * scale[-1] / scale[:-1]
                )
            quotient_direction = quotient_full[keep]
=======
            quotient_full = direction.clone()
            for row, column, fixed_index in zip(
                fixed_rows,
                fixed_columns,
                fixed_indices,
            ):
                row_start = row * d_model
                anchor_direction = direction[fixed_index].clone()
                quotient_full[
                    row_start : row_start + d_model
                ].sub_(
                    anchor_direction * scale[column] / scale
                )
            quotient_direction = quotient_full[keep]
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Recover the omitted key gradients from the LayerNorm-null directions.
    for parameter, ln_scale, d_model, fixed_rows in key_gauges:
        if parameter.grad is not None:
            fixed_indices = tuple(
                row * d_model + d_model - 1 for row in fixed_rows
            )
=======
    # Recover the omitted key gradients from the LayerNorm-null directions.
    for (
        parameter,
        ln_scale,
        d_model,
        fixed_rows,
        fixed_columns,
    ) in key_gauges:
        if parameter.grad is not None:
            fixed_indices = tuple(
                row * d_model + column
                for row, column in zip(fixed_rows, fixed_columns)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            scale = ln_scale.detach().reshape(-1).float()
            for row, fixed_index in zip(fixed_rows, fixed_indices):
                row_start = row * d_model
                omitted = -scale[-1] * (
                    virtual_grad[row_start:fixed_index] / scale[:-1]
                ).sum()
                total_sq.add_(omitted.pow(2))
=======
            scale = ln_scale.detach().reshape(-1).float()
            for row, column, fixed_index in zip(
                fixed_rows,
                fixed_columns,
                fixed_indices,
            ):
                row_start = row * d_model
                row_grad = virtual_grad[
                    row_start : row_start + d_model
                ]
                omitted = -scale[column] * (row_grad / scale).sum()
                total_sq.add_(omitted.pow(2))
>>>>>>> REPLACE

<<<<<<< SEARCH
    key_gauges = [
        (
            block.attn.qkv.weight,
            block.ln1.weight,
            model_cfg.d_model,
            block.attn.qkv.fixed_rows,
        )
        for block in model.blocks
    ]
=======
    key_gauges = [
        (
            block.attn.qkv.weight,
            block.ln1.weight,
            model_cfg.d_model,
            block.attn.qkv.fixed_rows,
            block.attn.qkv.fixed_columns,
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            *[parameter for parameter, _, _, _ in key_gauges],
=======
            *[
                parameter
                for parameter, _, _, _, _ in key_gauges
            ],
>>>>>>> REPLACE