MECHANISM: Independent per-head max-pivot value/output scale gauges

HYPOTHESIS: Normalizing both two-coordinate-gauged projection rows and absorbing their scales into the matching value rows will produce a 1499-parameter model with at least 99% accuracy.

INTENDED_EDIT: Replace both five-coordinate projection-row parameters with four-coordinate normalized charts, preserve their pivot indices, and use chart-aware reconstruction during training.

EVIDENCE: Reference Design 2 verified this exact construction at 99.64% accuracy with 1499 parameters; the preceding single-head scalar gauge also reached 99.94% at 1500 parameters.

<<<<<<< SEARCH
        # Downstream LayerNorms cancel the feature-uniform output coordinate.
        # Independent value/output basis gauges fix two relative coordinates
        # in the final projection row of each attention head.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.attn_drop = nn.Dropout(dropout)
=======
        # Downstream LayerNorms cancel the uniform output coordinate. Two
        # value/output shears are fixed in each head, and independent scalar
        # gauges normalize both heads' target rows.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.register_buffer(
            "proj_head_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "proj_last_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_relative = torch.cat(
            (
                self.proj_head_weight,
                self.proj_head_weight.new_zeros(2),
            )
        )
        last_relative = torch.cat(
            (
                self.proj_last_weight,
                self.proj_last_weight.new_zeros(2),
            )
        )
=======
        head_pivot = int(self.proj_head_pivot.item())
        head_chart = torch.cat(
            (
                self.proj_head_weight[:head_pivot],
                self.proj_head_weight.new_full((1,), 1.0),
                self.proj_head_weight[head_pivot:],
            )
        )
        head_free = head_chart * (
            (0.02 * math.sqrt(d_model - 3))
            / head_chart.norm()
        )
        last_pivot = int(self.proj_last_pivot.item())
        last_chart = torch.cat(
            (
                self.proj_last_weight[:last_pivot],
                self.proj_last_weight.new_full((1,), 1.0),
                self.proj_last_weight[last_pivot:],
            )
        )
        last_free = last_chart * (
            (0.02 * math.sqrt(d_model - 3))
            / last_chart.norm()
        )
        head_relative = torch.cat(
            (head_free, head_free.new_zeros(2))
        )
        last_relative = torch.cat(
            (last_free, last_free.new_zeros(2))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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

            # Apply the same conditioned two-coordinate gauge independently
            # in the second head.
            second_start = cfg.d_model - block.attn.head_dim
            second_target = cfg.d_model - 1
            second_pairs = [
                (left, right)
                for left in range(second_start, second_target)
                for right in range(left + 1, second_target)
            ]
            second_pivots = max(
                second_pairs,
                key=lambda pair: abs(
                    torch.linalg.det(
                        relative_proj_weight[list(pair), -2:]
                    ).item()
                ),
            )
            second_matrix = relative_proj_weight[
                list(second_pivots), -2:
            ].transpose(0, 1)
            second_coefficients = torch.linalg.solve(
                second_matrix,
                relative_proj_weight[second_target, -2:],
            )
            relative_proj_weight[second_target] = (
                relative_proj_weight[second_target]
                - (
                    second_coefficients.unsqueeze(1)
                    * relative_proj_weight[list(second_pivots)]
                ).sum(dim=0)
            )
            for pivot, coefficient in zip(
                second_pivots, second_coefficients
            ):
                full_qkv_weight[value_start + pivot] = (
                    full_qkv_weight[value_start + pivot]
                    + coefficient
                    * full_qkv_weight[value_start + second_target]
                )
=======
            target_rows = (
                block.attn.head_dim - 1,
                cfg.d_model - 1,
            )

            for target_row in target_rows:
                head_start = (
                    target_row - block.attn.head_dim + 1
                )
                pivot_pairs = [
                    (left, right)
                    for left in range(head_start, target_row)
                    for right in range(left + 1, target_row)
                ]
                pivots = max(
                    pivot_pairs,
                    key=lambda pair: abs(
                        torch.linalg.det(
                            relative_proj_weight[list(pair), -2:]
                        ).item()
                    ),
                )
                matrix = relative_proj_weight[
                    list(pivots), -2:
                ].transpose(0, 1)
                coefficients = torch.linalg.solve(
                    matrix,
                    relative_proj_weight[target_row, -2:],
                )
                relative_proj_weight[target_row] = (
                    relative_proj_weight[target_row]
                    - (
                        coefficients.unsqueeze(1)
                        * relative_proj_weight[list(pivots)]
                    ).sum(dim=0)
                )
                for pivot, coefficient in zip(
                    pivots, coefficients
                ):
                    full_qkv_weight[value_start + pivot] = (
                        full_qkv_weight[value_start + pivot]
                        + coefficient
                        * full_qkv_weight[value_start + target_row]
                    )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                relative_proj_weight[-1, :-2]
            )
=======
            first_target = block.attn.head_dim - 1
            head_free = relative_proj_weight[first_target, :-2]
            head_pivot = int(head_free.abs().argmax().item())
            pivot_value = head_free[head_pivot]
            head_chart = head_free / pivot_value
            gauge_norm = 0.02 * math.sqrt(head_free.numel())
            head_scale = (
                pivot_value.sign() * head_free.norm() / gauge_norm
            )
            full_qkv_weight[value_start + first_target] = (
                head_scale
                * full_qkv_weight[value_start + first_target]
            )
            block.attn.proj_head_pivot.fill_(head_pivot)

            last_target = cfg.d_model - 1
            last_free = relative_proj_weight[last_target, :-2]
            last_pivot = int(last_free.abs().argmax().item())
            last_pivot_value = last_free[last_pivot]
            last_chart = last_free / last_pivot_value
            last_gauge_norm = 0.02 * math.sqrt(last_free.numel())
            last_scale = (
                last_pivot_value.sign()
                * last_free.norm()
                / last_gauge_norm
            )
            full_qkv_weight[value_start + last_target] = (
                last_scale
                * full_qkv_weight[value_start + last_target]
            )
            block.attn.proj_last_pivot.fill_(last_pivot)

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
                torch.cat(
                    (
                        head_chart[:head_pivot],
                        head_chart[head_pivot + 1:],
                    )
                )
            )
            block.attn.proj_last_weight = nn.Parameter(
                torch.cat(
                    (
                        last_chart[:last_pivot],
                        last_chart[last_pivot + 1:],
                    )
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
def reconstruct_attention_output_weight(
    weight, head_weight, last_weight, head_dim
):
    head_relative = torch.cat(
        (head_weight, head_weight.new_zeros(2))
    )
    last_relative = torch.cat(
        (last_weight, last_weight.new_zeros(2))
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
=======
def reconstruct_attention_output_weight(
    weight,
    head_weight,
    last_weight,
    head_dim,
    head_pivot,
    last_pivot,
):
    head_chart = torch.cat(
        (
            head_weight[:head_pivot],
            head_weight.new_full((1,), 1.0),
            head_weight[head_pivot:],
        )
    )
    head_free = head_chart * (
        (0.02 * math.sqrt(head_weight.numel() + 1))
        / head_chart.norm()
    )
    last_chart = torch.cat(
        (
            last_weight[:last_pivot],
            last_weight.new_full((1,), 1.0),
            last_weight[last_pivot:],
        )
    )
    last_free = last_chart * (
        (0.02 * math.sqrt(last_weight.numel() + 1))
        / last_chart.norm()
    )
    head_relative = torch.cat(
        (head_free, head_free.new_zeros(2))
    )
    last_relative = torch.cat(
        (last_free, last_free.new_zeros(2))
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
=======
        value_bias_grads = []
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            head_pivot,
            last_pivot,
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
                        head_pivot,
                        last_pivot,
                    )
                )
                grad = (
                    full_proj_weight
                    * full_proj_grad.unsqueeze(1)
                ).sum(dim=0)
            value_bias_grads.append(grad)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
=======
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            head_pivot,
            last_pivot,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_delta = (
                reconstruct_attention_output_weight(
                    proj_weight,
                    proj_head_weight,
                    proj_last_weight,
                    head_dim,
                )
                @ omitted_value
            )
=======
            full_delta = (
                reconstruct_attention_output_weight(
                    proj_weight,
                    proj_head_weight,
                    proj_last_weight,
                    head_dim,
                    head_pivot,
                    last_pivot,
                )
                @ omitted_value
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    for (
        qkv_bias,
        proj_weight,
        proj_head_weight,
        proj_last_weight,
        proj_bias,
        head_dim,
        head_pivot,
        last_pivot,
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
            head_pivot,
            last_pivot,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.proj.bias,
            block.attn.head_dim,
        )
        for block in model.blocks
=======
            block.attn.proj.bias,
            block.attn.head_dim,
            int(block.attn.proj_head_pivot.item()),
            int(block.attn.proj_last_pivot.item()),
        )
        for block in model.blocks
>>>>>>> REPLACE