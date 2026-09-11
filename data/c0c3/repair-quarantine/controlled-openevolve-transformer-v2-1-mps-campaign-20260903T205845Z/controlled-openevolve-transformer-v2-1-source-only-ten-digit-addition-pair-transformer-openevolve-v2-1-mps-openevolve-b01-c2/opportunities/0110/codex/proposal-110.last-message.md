MECHANISM: Dense-AdamW optimization of the sixth attention-LayerNorm column-scale quotient

HYPOTHESIS: Extending the qualified 1,533-parameter all-row-QKV design by fixing LayerNorm scale coordinate 0, while maintaining its omitted dense AdamW moment and absorbing each update into QKV column 0, will produce a 1,532-parameter model with at least 99% accuracy.

INTENDED_EDIT: Gauge second-head query row 6 using the verified dense-row optimizer, anchor LayerNorm scale coordinate 0, reconstruct the full scale vector explicitly, and train the removed scale virtually through an equivalent multiplicative QKV-column update.

EVIDENCE: The all-row-QKV design reached 99.6% at 1,533 parameters, while the direct coordinate-0 scale anchor reached 98.22%—substantially closer than coordinate 1—and prior dense-coordinate optimization rescued sensitive QKV quotient training, motivating the same optimizer treatment for this exact scale quotient.

<<<<<<< SEARCH
        # Gauge the final second-head query row while retaining every verified
        # key and value gauge. Sensitive rows 15, 20, and 23 use recovered
        # dense-coordinate AdamW moments.
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 3,
=======
        # Gauge every QKV row. The newly included second-head query row 6 and
        # sensitive rows 15, 20, and 23 use recovered dense-coordinate AdamW
        # moments.
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 2,
            head_dim + 3,
>>>>>>> REPLACE

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
        return F.layer_norm(x, self.normalized_shape, weight, None, 1e-5)
=======
class AnchoredScaleLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        # Coordinate 0 and coordinates 3–7 are fixed; coordinates 1–2 remain
        # learned. The omitted coordinate-0 update is absorbed by QKV.
        self.weight = nn.Parameter(torch.ones(d_model - 6))

    def dense_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.weight.new_ones(1),
                self.weight,
                self.weight.new_ones(5),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, self.normalized_shape, self.dense_weight(), None, 1e-5
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        anchored_weight = torch.stack(anchored_rows)
        reduced_grad = torch.stack(reduced_grads)
        scales = torch.cat(
            (
                normalization.weight,
                normalization.weight.new_ones(
                    module.in_features - normalization.weight.numel()
                ),
            )
        )
        null_direction = scales.reciprocal()
=======
        anchored_weight = torch.stack(anchored_rows)
        reduced_grad = torch.stack(reduced_grads)

        # Recover the gradient of omitted LayerNorm scale coordinate 0. With
        # that scale represented by one, dL/d(scale) is the inner product of
        # QKV column 0 and its chart gradient.
        chart_weight = parameter.view(module.out_features, row_width)
        chart_grad = parameter.grad.view(module.out_features, row_width)
        fixed_scale_grad = (
            chart_weight[:, 0] * chart_grad[:, 0]
        ).sum()

        scales = normalization.dense_weight()
        null_direction = scales.reciprocal()
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Update key row 15 and value rows 20 and 23 in eight-coordinate
        # geometry, then project them back into the anchored chart.
=======
        # Update query row 6, key row 15, and value rows 20 and 23 in
        # eight-coordinate geometry, then return to the anchored chart.
>>>>>>> REPLACE

<<<<<<< SEARCH
        for dense_row, (start, stop) in zip(
            dense_weight, target_slices
        ):
            parameter[start:stop].copy_(dense_row[:-1])
=======
        for dense_row, (start, stop) in zip(
            dense_weight, target_slices
        ):
            parameter[start:stop].copy_(dense_row[:-1])

        # Give the omitted coordinate-0 LayerNorm scale its own dense AdamW
        # moments, then restore its fixed value of one by absorbing the update
        # into the corresponding QKV column.
        scale_exp_avg = state["scale_exp_avg"]
        scale_exp_avg_sq = state["scale_exp_avg_sq"]
        scale_exp_avg.mul_(beta1).add_(
            fixed_scale_grad, alpha=1.0 - beta1
        )
        scale_exp_avg_sq.mul_(beta2).add_(
            fixed_scale_grad.square(), alpha=1.0 - beta2
        )
        scale_denom = scale_exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        fixed_scale = parameter.new_ones(())
        fixed_scale.mul_(1.0 - lr * weight_decay)
        fixed_scale.add_(
            scale_exp_avg / scale_denom,
            alpha=-lr / bias_correction1,
        )
        parameter.view(module.out_features, row_width)[:, 0].mul_(
            fixed_scale
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Residual-output quotients retain dense output-coordinate moments. Key
    # row 15 and value rows 20 and 23 retain their verified dense-coordinate
    # updates; query rows 2–3 and 7 and value rows 21–22 use the reduced chart.
=======
    # Residual-output quotients retain dense output-coordinate moments. Query
    # row 6 and sensitive rows 15, 20, and 23 use dense row moments, while the
    # omitted LayerNorm scale coordinate 0 receives a virtual dense update.
>>>>>>> REPLACE

<<<<<<< SEARCH
                (
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                    3 * qkv.in_features - 1,
                ),
=======
                (
                    block.attn.head_dim + 2,
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                    3 * qkv.in_features - 1,
                ),
>>>>>>> REPLACE

<<<<<<< SEARCH
                "dense_exp_avg_sq": parameter.new_zeros(
                    len(target_rows), module.in_features
                ),
=======
                "dense_exp_avg_sq": parameter.new_zeros(
                    len(target_rows), module.in_features
                ),
                "scale_exp_avg": parameter.new_zeros(()),
                "scale_exp_avg_sq": parameter.new_zeros(()),
>>>>>>> REPLACE