MECHANISM: Additional within-head value/output basis gauge fixing

HYPOTHESIS: Fixing a second relative attention-output coordinate in the first head will reduce the verified 1503-parameter design to 1502 parameters while maintaining at least 99% accuracy, because two conditioned pivot directions can select this representative through an invertible value/output basis change without changing the initialized function.

INTENDED_EDIT: Reproduce the verified per-head projection gauges, then store five coordinates for the first head’s gauged projection row and six for the second head’s row, reconstructing the omitted coordinates as zero and applying matching value-weight basis changes at initialization.

EVIDENCE: Reference Design 1 achieved 99.90% accuracy at 1503 parameters after independently fixing one attention-output coordinate in each head; extending that successful value/output symmetry by one coordinate is the smallest symmetry-grounded reduction, while prior additional query-bias restrictions failed.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Within each head, an invertible query/key basis change
        # can additionally fix one generic query-bias coordinate to zero.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Downstream LayerNorms cancel its feature-uniform coordinate.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. A query/key basis gauge fixes the final query-bias
        # coordinate.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Downstream LayerNorms cancel the feature-uniform output coordinate.
        # Value/output basis gauges additionally fix two relative coordinates
        # in one head's projection row and one in the other head's row.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight_relative = torch.cat(
            (
                self.proj.weight,
                self.proj.weight.new_zeros(
                    (self.proj.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        proj_weight = (
            weight_relative
            + self.proj.weight.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
=======
        head_relative = torch.cat(
            (
                self.proj_head_weight,
                self.proj_head_weight.new_zeros(2),
            )
        )
        last_relative = torch.cat(
            (
                self.proj_last_weight,
                self.proj_last_weight.new_zeros(1),
            )
        )
        split = self.head_dim - 1
        weight_rows = torch.cat(
            (
                self.proj.weight[:split],
                head_relative.unsqueeze(0),
                self.proj.weight[split:],
                last_relative.unsqueeze(0),
            ),
            dim=0,
        )
        weight_relative = torch.cat(
            (
                weight_rows,
                weight_rows.new_zeros((weight_rows.size(0), 1)),
            ),
            dim=-1,
        )
        proj_weight = (
            weight_relative
            + weight_rows.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.blocks:
            full_qkv_weight = block.attn.qkv.weight.detach()
            block.attn.qkv.weight = nn.Parameter(
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1)
            )
            block.attn.proj.weight = nn.Parameter(
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )
            full_fc1_weight = block.mlp.fc1.weight.detach()
            block.mlp.fc1.weight = nn.Parameter(
                full_fc1_weight[:, :-1] - full_fc1_weight[:, -1:]
            )
            full_fc2_weight = (
                block.mlp.fc2.weight.detach().transpose(0, 1)
            )
            block.mlp.fc2.weight = nn.Parameter(
                full_fc2_weight[:, :-1] - full_fc2_weight[:, -1:]
            )
=======
        for block in self.blocks:
            full_qkv_weight = block.attn.qkv.weight.detach().clone()
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1).clone()
            )
            relative_proj_weight = (
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )
            value_start = 2 * cfg.d_model

            # Use two well-conditioned pivot directions to make the final
            # value row of the first head zero in two relative coordinates.
            first_target = block.attn.head_dim - 1
            first_pairs = [
                (left, right)
                for left in range(first_target)
                for right in range(left + 1, first_target)
            ]
            first_pivots = max(
                first_pairs,
                key=lambda pair: abs(
                    torch.linalg.det(
                        relative_proj_weight[list(pair), -2:]
                    ).item()
                ),
            )
            first_matrix = relative_proj_weight[
                list(first_pivots), -2:
            ].transpose(0, 1)
            first_coefficients = torch.linalg.solve(
                first_matrix,
                relative_proj_weight[first_target, -2:],
            )
            relative_proj_weight[first_target] = (
                relative_proj_weight[first_target]
                - (
                    first_coefficients.unsqueeze(1)
                    * relative_proj_weight[list(first_pivots)]
                ).sum(dim=0)
            )
            for pivot, coefficient in zip(
                first_pivots, first_coefficients
            ):
                full_qkv_weight[value_start + pivot] = (
                    full_qkv_weight[value_start + pivot]
                    + coefficient
                    * full_qkv_weight[value_start + first_target]
                )

            # Retain the qualified one-coordinate gauge in the second head.
            second_start = cfg.d_model - block.attn.head_dim
            second_target = cfg.d_model - 1
            second_pivot = second_start + int(
                relative_proj_weight[
                    second_start:second_target, -1
                ].abs().argmax().item()
            )
            second_coefficient = (
                relative_proj_weight[second_target, -1]
                / relative_proj_weight[second_pivot, -1]
            )
            relative_proj_weight[second_target] = (
                relative_proj_weight[second_target]
                - second_coefficient
                * relative_proj_weight[second_pivot]
            )
            full_qkv_weight[value_start + second_pivot] = (
                full_qkv_weight[value_start + second_pivot]
                + second_coefficient
                * full_qkv_weight[value_start + second_target]
            )

            block.attn.qkv.weight = nn.Parameter(
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.proj.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_proj_weight[:first_target],
                        relative_proj_weight[
                            first_target + 1:-1
                        ],
                    ),
                    dim=0,
                )
            )
            block.attn.proj_head_weight = nn.Parameter(
                relative_proj_weight[first_target, :-2]
            )
            block.attn.proj_last_weight = nn.Parameter(
                relative_proj_weight[-1, :-1]
            )

            full_fc1_weight = block.mlp.fc1.weight.detach()
            block.mlp.fc1.weight = nn.Parameter(
                full_fc1_weight[:, :-1] - full_fc1_weight[:, -1:]
            )
            full_fc2_weight = (
                block.mlp.fc2.weight.detach().transpose(0, 1)
            )
            block.mlp.fc2.weight = nn.Parameter(
                full_fc2_weight[:, :-1] - full_fc2_weight[:, -1:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
def reconstruct_output_weight(param):
    return reconstruct_input_weight(param).transpose(0, 1)
=======
def reconstruct_attention_output_weight(
    weight, head_weight, last_weight, head_dim
):
    head_relative = torch.cat(
        (head_weight, head_weight.new_zeros(2))
    )
    last_relative = torch.cat(
        (last_weight, last_weight.new_zeros(1))
    )
    split = head_dim - 1
    rows = torch.cat(
        (
            weight[:split],
            head_relative.unsqueeze(0),
            weight[split:],
            last_relative.unsqueeze(0),
        ),
        dim=0,
    )
    relative = torch.cat(
        (rows, rows.new_zeros((rows.size(0), 1))),
        dim=-1,
    )
    return (
        relative + rows.mean(dim=-1, keepdim=True)
    ).transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias_grads = []
        for qkv_bias, proj_weight, proj_bias in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                full_proj_grad = torch.cat(
                    (
                        proj_bias.grad.detach(),
                        -proj_bias.grad.detach().sum().view(1),
                    )
                )
                full_proj_weight = reconstruct_output_weight(
                    proj_weight.detach()
                )
                grad = (
                    full_proj_weight
                    * full_proj_grad.unsqueeze(1)
                ).sum(dim=0)
            value_bias_grads.append(grad)
=======
        value_bias_grads = []
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
        ) in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                full_proj_grad = torch.cat(
                    (
                        proj_bias.grad.detach(),
                        -proj_bias.grad.detach().sum().view(1),
                    )
                )
                full_proj_weight = (
                    reconstruct_attention_output_weight(
                        proj_weight.detach(),
                        proj_head_weight.detach(),
                        proj_last_weight.detach(),
                        head_dim,
                    )
                )
                grad = (
                    full_proj_weight
                    * full_proj_grad.unsqueeze(1)
                ).sum(dim=0)
            value_bias_grads.append(grad)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (qkv_bias, proj_weight, proj_bias), grad in zip(
            self.value_bias_specs, value_bias_grads
        ):
            if grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(candidate is qkv_bias for candidate in group["params"])
            )
            if group["maximize"]:
                grad = -grad

            state = self.state[qkv_bias]
            if "value_quotient_step" not in state:
                state["value_quotient_step"] = 0
                state["value_quotient_exp_avg"] = qkv_bias.new_zeros(
                    grad.shape
                )
                state["value_quotient_exp_avg_sq"] = qkv_bias.new_zeros(
                    grad.shape
                )

            state["value_quotient_step"] += 1
            step = state["value_quotient_step"]
            beta1, beta2 = group["betas"]
            exp_avg = state["value_quotient_exp_avg"]
            exp_avg_sq = state["value_quotient_exp_avg_sq"]

            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                grad, grad, value=1.0 - beta2
            )

            step_size = group["lr"] / (1.0 - beta1 ** step)
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(1.0 - beta2 ** step)
            ).add_(group["eps"])
            omitted_value = -step_size * exp_avg / denom
            full_delta = (
                reconstruct_output_weight(proj_weight)
                @ omitted_value
            )
=======
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
            if grad is None:
                continue

            group = next(
                group
                for group in self.param_groups
                if any(candidate is qkv_bias for candidate in group["params"])
            )
            if group["maximize"]:
                grad = -grad

            state = self.state[qkv_bias]
            if "value_quotient_step" not in state:
                state["value_quotient_step"] = 0
                state["value_quotient_exp_avg"] = qkv_bias.new_zeros(
                    grad.shape
                )
                state["value_quotient_exp_avg_sq"] = qkv_bias.new_zeros(
                    grad.shape
                )

            state["value_quotient_step"] += 1
            step = state["value_quotient_step"]
            beta1, beta2 = group["betas"]
            exp_avg = state["value_quotient_exp_avg"]
            exp_avg_sq = state["value_quotient_exp_avg_sq"]

            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                grad, grad, value=1.0 - beta2
            )

            step_size = group["lr"] / (1.0 - beta1 ** step)
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(1.0 - beta2 ** step)
            ).add_(group["eps"])
            omitted_value = -step_size * exp_avg / denom
            full_delta = (
                reconstruct_attention_output_weight(
                    proj_weight,
                    proj_head_weight,
                    proj_last_weight,
                    head_dim,
                )
                @ omitted_value
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for qkv_bias, proj_weight, proj_bias in value_bias_specs:
        if proj_bias.grad is None:
            continue
        full_proj_grad = torch.cat(
            (
                proj_bias.grad.detach(),
                -proj_bias.grad.detach().sum().view(1),
            )
        )
        full_proj_weight = reconstruct_output_weight(
            proj_weight.detach()
        )
        omitted_grad = (
            full_proj_weight
            * full_proj_grad.unsqueeze(1)
        ).sum(dim=0)
=======
    for (
        qkv_bias,
        proj_weight,
        proj_head_weight,
        proj_last_weight,
        proj_bias,
        head_dim,
    ) in value_bias_specs:
        if proj_bias.grad is None:
            continue
        full_proj_grad = torch.cat(
            (
                proj_bias.grad.detach(),
                -proj_bias.grad.detach().sum().view(1),
            )
        )
        full_proj_weight = reconstruct_attention_output_weight(
            proj_weight.detach(),
            proj_head_weight.detach(),
            proj_last_weight.detach(),
            head_dim,
        )
        omitted_grad = (
            full_proj_weight
            * full_proj_grad.unsqueeze(1)
        ).sum(dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj.bias,
        )
        for block in model.blocks
    ]
=======
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_head_weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE