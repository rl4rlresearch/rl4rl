MECHANISM: Reduced-chart final second-head value-row quotient

HYPOTHESIS: Extending the verified 1,540-parameter design to second-head value row 23 with ordinary reduced-coordinate AdamW will produce a 1,539-parameter model with at least 99% accuracy.

INTENDED_EDIT: Gauge every key and value row, retaining dense-coordinate AdamW only for sensitive key row 15 and value row 20 while value rows 21–23 use ordinary reduced-coordinate updates.

EVIDENCE: Value row 22 achieved 99.62% at 1,540 parameters with reduced-coordinate AdamW, and row 21 similarly improved from 98.28% with dense-coordinate training to 99.93% with reduced-coordinate training; this motivates applying the same geometry to adjacent final value row 23.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges, gauge every first-head key
        # row plus three adjacent second-head key rows, and gauge every
        # first-head value row.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + head_dim,
            d_model + head_dim + 1,
            d_model + head_dim + 2,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
        )
=======
        # Retain the balanced query gauges and gauge every key and value row.
        # Sensitive key row 15 and value row 20 use dense-coordinate updates;
        # value rows 21–23 use ordinary reduced-coordinate AdamW.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + head_dim,
            d_model + head_dim + 1,
            d_model + head_dim + 2,
            d_model + head_dim + 3,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
            2 * d_model + head_dim,
            2 * d_model + head_dim + 1,
            2 * d_model + head_dim + 2,
            2 * d_model + head_dim + 3,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        parameter.add_(
            module.output_basis.transpose(0, 1) @ dense_update,
            alpha=-lr / bias_correction1,
        )


def save_json(path: Path, obj: Dict) -> None:
=======
        parameter.add_(
            module.output_basis.transpose(0, 1) @ dense_update,
            alpha=-lr / bias_correction1,
        )


@torch.no_grad()
def step_dense_qkv_row_quotients(
    specifications, states, lr: float, weight_decay: float
) -> None:
    """Apply dense AdamW moments to selected normalized-input row quotients."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for (module, normalization, target_rows), state in zip(
        specifications, states
    ):
        parameter = module.weight
        if parameter.grad is None:
            continue

        row_width = module.in_features - 1
        target_slices = []
        anchored_rows = []
        reduced_grads = []
        for target_row in target_rows:
            target_index = module.gauged_rows.index(target_row)
            start = target_index * row_width
            stop = start + row_width
            target_slices.append((start, stop))
            anchored_rows.append(parameter[start:stop].clone())
            reduced_grads.append(parameter.grad[start:stop])

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
        last_grad = -(
            reduced_grad * null_direction[:-1]
        ).sum(dim=1, keepdim=True) / null_direction[-1]
        dense_grad = torch.cat((reduced_grad, last_grad), dim=1)

        state["step"] += 1
        step = state["step"]
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        # Update all coefficients in the reduced chart first. This preserves
        # the verified ordinary AdamW geometry for value rows 21 through 23.
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(parameter.grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            parameter.grad, parameter.grad, value=1.0 - beta2
        )
        denom = exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        parameter.mul_(1.0 - lr * weight_decay)
        parameter.add_(
            exp_avg / denom,
            alpha=-lr / bias_correction1,
        )

        # Replace key row 15 and value row 20 with projected updates computed
        # using moments in their original eight-coordinate geometry.
        dense_exp_avg = state["dense_exp_avg"]
        dense_exp_avg_sq = state["dense_exp_avg_sq"]
        dense_exp_avg.mul_(beta1).add_(dense_grad, alpha=1.0 - beta1)
        dense_exp_avg_sq.mul_(beta2).addcmul_(
            dense_grad, dense_grad, value=1.0 - beta2
        )
        dense_denom = dense_exp_avg_sq.sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(eps)
        dense_weight = torch.cat(
            (
                anchored_weight,
                anchored_weight.new_zeros(len(target_rows), 1),
            ),
            dim=1,
        )
        dense_weight.mul_(1.0 - lr * weight_decay)
        dense_weight.add_(
            dense_exp_avg / dense_denom,
            alpha=-lr / bias_correction1,
        )
        dense_weight[:, :-1].sub_(
            dense_weight[:, -1:]
            * null_direction[:-1]
            / null_direction[-1]
        )
        for dense_row, (start, stop) in zip(
            dense_weight, target_slices
        ):
            parameter[start:stop].copy_(dense_row[:-1])


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve the successful ordinary quotient optimizer for attention bias.
    # Newly gauged weights and the sensitive final-MLP bias use moments in the
    # original dense output coordinates before each update is projected.
    quotient_specifications = []
    for block in model.blocks:
        quotient_specifications.extend(
            (
                (block.attn.proj, "weight"),
                (block.mlp.fc2, "weight"),
                (block.mlp.fc2, "bias"),
            )
        )
    quotient_parameter_ids = {
        id(getattr(module, parameter_name))
        for module, parameter_name in quotient_specifications
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quotient_parameter_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    quotient_states = []
=======
    # Residual-output quotients retain dense output-coordinate moments. Key
    # row 15 and value row 20 retain their verified dense-coordinate updates;
    # newly gauged value row 23 follows rows 21–22 in the reduced chart.
    quotient_specifications = []
    qkv_row_specifications = []
    for block in model.blocks:
        quotient_specifications.extend(
            (
                (block.attn.proj, "weight"),
                (block.mlp.fc2, "weight"),
                (block.mlp.fc2, "bias"),
            )
        )
        qkv = block.attn.qkv
        qkv_row_specifications.append(
            (
                qkv,
                block.ln1,
                (
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                ),
            )
        )
    quotient_parameter_ids = {
        id(getattr(module, parameter_name))
        for module, parameter_name in quotient_specifications
    }
    qkv_row_parameter_ids = {
        id(module.weight)
        for module, _, _ in qkv_row_specifications
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quotient_parameter_ids
            and id(parameter) not in qkv_row_parameter_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    qkv_row_states = []
    for module, _, target_rows in qkv_row_specifications:
        parameter = module.weight
        qkv_row_states.append(
            {
                "step": 0,
                "exp_avg": parameter.new_zeros(parameter.shape),
                "exp_avg_sq": parameter.new_zeros(parameter.shape),
                "dense_exp_avg": parameter.new_zeros(
                    len(target_rows), module.in_features
                ),
                "dense_exp_avg_sq": parameter.new_zeros(
                    len(target_rows), module.in_features
                ),
            }
        )
    quotient_states = []
>>>>>>> REPLACE

<<<<<<< SEARCH
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        step_dense_output_quotients(
=======
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        step_dense_qkv_row_quotients(
            qkv_row_specifications,
            qkv_row_states,
            lr_now,
            train_cfg.weight_decay,
        )
        optimizer.step()
        step_dense_output_quotients(
>>>>>>> REPLACE