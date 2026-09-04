MECHANISM: Dense-AdamW second-head value-row quotient extension

HYPOTHESIS: Extending the verified 1,542-parameter design to second-head value row 21, while retaining dense-coordinate AdamW moments for key row 15 and value rows 20–21, will produce a 1,541-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce every gauge and optimizer from the 1,542-parameter reference, add the adjacent second-head value-row gauge at row 21, and train all optimization-sensitive QKV quotients through projected eight-coordinate AdamW updates.

EVIDENCE: The first second-head value-row extension reached 100% accuracy at 1,542 parameters when trained with recovered dense-coordinate moments; row 21 is the nearest untested continuation of that successful pattern.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges, gauge every first-head key
        # row, and gauge every first-head value row.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
        )
=======
        # Retain the verified balanced query gauges, gauge every key row,
        # gauge every first-head value row, and extend the dense-coordinate
        # quotient through the first two value rows of the second head.
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
    # Use dense-coordinate AdamW moments for quotiented residual weights and
    # the optimization-sensitive final-MLP bias.
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
    # Residual-output quotients retain dense output-coordinate moments. The
    # final key row and first two second-head value rows additionally recover
    # moments for their omitted normalized-input coefficients.
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
                    2 * qkv.in_features + block.attn.head_dim + 1,
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