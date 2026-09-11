MECHANISM: Dense-AdamW final key-row quotient

HYPOTHESIS: Gauging both remaining second-head key rows while training row 15 with recovered eight-coordinate AdamW moments will produce a 1,543-parameter model with at least 99% accuracy.

INTENDED_EDIT: Add key rows 14–15 to the normalized-input gauge and project a dense-coordinate AdamW update for row 15 back into its anchored seven-parameter chart.

EVIDENCE: Row 14 achieved 99.85% at 1,544 parameters, whereas row 15 collapsed to 90.92%; dense-coordinate AdamW previously rescued an exact quotient from 98.89% to 99.97%, implicating optimizer geometry rather than capacity.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges, gauge every first-head key
        # row plus two adjacent second-head key rows, and gauge every
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
            2 * d_model,
=======
        # Retain the verified balanced query gauges, gauge every key row in
        # both heads, and gauge every first-head value row.
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
def step_dense_last_key_quotients(
    specifications, states, lr: float, weight_decay: float
) -> None:
    """Use dense AdamW moments for the final key-row quotient."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for (module, normalization), state in zip(specifications, states):
        parameter = module.weight
        if parameter.grad is None:
            continue

        row_width = module.in_features - 1
        target_row = 2 * module.in_features - 1
        target_index = module.gauged_rows.index(target_row)
        start = target_index * row_width
        stop = start + row_width

        anchored_weight = parameter[start:stop].clone()
        reduced_grad = parameter.grad[start:stop]

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
            null_direction[:-1] * reduced_grad
        ).sum() / null_direction[-1]
        dense_grad = torch.cat((reduced_grad, last_grad.unsqueeze(0)))

        state["step"] += 1
        step = state["step"]
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        # Preserve ordinary reduced-coordinate AdamW for every other QKV
        # coefficient.
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

        # Replace row 15's reduced update with an ordinary eight-coordinate
        # AdamW update, then return to the zero-final-coefficient chart.
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
            (anchored_weight, anchored_weight.new_zeros(1))
        )
        dense_weight.mul_(1.0 - lr * weight_decay)
        dense_weight.add_(
            dense_exp_avg / dense_denom,
            alpha=-lr / bias_correction1,
        )
        dense_weight[:-1].sub_(
            dense_weight[-1]
            * null_direction[:-1]
            / null_direction[-1]
        )
        parameter[start:stop].copy_(dense_weight[:-1])


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Attention bias retains its successful ordinary quotient optimizer.
    # Newly gauged weights and the sensitive final-MLP bias use moments in
    # their original dense output coordinates before projection.
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
=======
    # Residual-output quotients retain dense output-coordinate moments. The
    # newly completed key gauge additionally recovers dense moments for its
    # omitted normalized-input coefficient.
    quotient_specifications = []
    last_key_specifications = []
    for block in model.blocks:
        quotient_specifications.extend(
            (
                (block.attn.proj, "weight"),
                (block.mlp.fc2, "weight"),
                (block.mlp.fc2, "bias"),
            )
        )
        last_key_specifications.append((block.attn.qkv, block.ln1))
    quotient_parameter_ids = {
        id(getattr(module, parameter_name))
        for module, parameter_name in quotient_specifications
    }
    last_key_parameter_ids = {
        id(module.weight) for module, _ in last_key_specifications
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in quotient_parameter_ids
            and id(parameter) not in last_key_parameter_ids
        ],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    last_key_states = []
    for module, _ in last_key_specifications:
        parameter = module.weight
        last_key_states.append(
            {
                "step": 0,
                "exp_avg": parameter.new_zeros(parameter.shape),
                "exp_avg_sq": parameter.new_zeros(parameter.shape),
                "dense_exp_avg": parameter.new_zeros(module.in_features),
                "dense_exp_avg_sq": parameter.new_zeros(module.in_features),
            }
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.step()
        step_dense_output_quotients(
=======
        step_dense_last_key_quotients(
            last_key_specifications,
            last_key_states,
            lr_now,
            train_cfg.weight_decay,
        )
        optimizer.step()
        step_dense_output_quotients(
>>>>>>> REPLACE