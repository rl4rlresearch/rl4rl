MECHANISM: Balanced per-head value/output basis gauge fixing

HYPOTHESIS: Fixing a second relative attention-output coordinate in the second head will reduce the qualified 1502-parameter design to 1501 parameters while maintaining at least 99% accuracy, because each head admits an independent invertible value/output basis change and a well-conditioned two-pivot initialization preserves the represented function.

INTENDED_EDIT: Reproduce the qualified projection gauges, but store five coordinates for the final projection row of each head, reconstruct two omitted coordinates per row as zero, and apply matching within-head value-weight basis changes at initialization.

EVIDENCE: The 1502-parameter design achieved 99.99% accuracy with two fixed projection coordinates in the first head and one in the second; extending the same successful two-coordinate construction to the second head is the smallest symmetry-grounded reduction and avoids the query-bias restrictions that previously failed.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        # Downstream LayerNorms cancel its feature-uniform coordinate.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model)
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
                self.proj_last_weight.new_zeros(2),
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
        # LayerNorm-null input coordinates and final-LayerNorm-null output
        # coordinates are retained only through relative representatives.
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
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. In each head, use two well-conditioned
        # pivot directions to fix two projection coordinates while applying
        # the inverse value-weight basis change.
        for block in self.blocks:
            full_qkv_weight = block.attn.qkv.weight.detach().clone()
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1).clone()
            )
            relative_proj_weight = (
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )
            value_start = 2 * cfg.d_model
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

            block.attn.qkv.weight = nn.Parameter(
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            first_target = block.attn.head_dim - 1
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
        # Track each omitted value-bias coordinate with full-coordinate
        # AdamW, then return to the zero gauge by folding its update into
        # the already-updated attention projection bias.
        for (qkv_bias, proj_weight, proj_bias), grad in zip(
            self.value_bias_specs, value_bias_grads
        ):
=======
        # Track each omitted value-bias coordinate with full-coordinate
        # AdamW, then return to the zero gauge by folding its update into
        # the already-updated attention projection bias.
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_delta = (
                reconstruct_output_weight(proj_weight)
                @ omitted_value
            )
=======
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