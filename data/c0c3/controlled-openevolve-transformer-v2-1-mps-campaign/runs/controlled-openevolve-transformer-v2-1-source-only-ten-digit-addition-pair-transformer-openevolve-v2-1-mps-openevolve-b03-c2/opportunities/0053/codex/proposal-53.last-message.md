MECHANISM: Secondary-row scalar value/output basis gauge

HYPOTHESIS: Fixing one disjoint relative projection coordinate in another row of the first attention head will reduce the model from 1501 to 1500 parameters while retaining at least 99% accuracy, because a best-pivot scalar shear preserves the initialized function without the three-coordinate same-row conditioning that reached only 98.82%.

INTENDED_EDIT: Add a six-coordinate auxiliary projection-row parameter, reconstruct its omitted third-from-last relative coordinate as zero, and initialize it through a function-preserving one-pivot value/output basis change.

EVIDENCE: The balanced two-coordinate-per-head design achieved 100% at 1501 parameters, while extending one row to three simultaneously fixed coordinates scored 98.82%; this tests an unused, independently conditioned scalar gauge on a different row.

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
=======
        # Downstream LayerNorms cancel the feature-uniform output coordinate.
        # Independent value/output basis gauges fix two relative coordinates
        # in each head's final row and one disjoint coordinate in another row.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_aux_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
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
=======
        aux_relative = torch.cat(
            (
                self.proj_aux_weight[:-2],
                self.proj_aux_weight.new_zeros(1),
                self.proj_aux_weight[-2:],
            )
        )
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
        split = self.head_dim - 2
        weight_rows = torch.cat(
            (
                self.proj.weight[:split],
                aux_relative.unsqueeze(0),
                head_relative.unsqueeze(0),
                self.proj.weight[split:],
                last_relative.unsqueeze(0),
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. In each head, use two well-conditioned
        # pivot directions to fix two projection coordinates while applying
        # the inverse value-weight basis change.
=======
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. Use two-pivot gauges on the final row of
        # each head plus a scalar, disjoint-coordinate gauge on another row.
>>>>>>> REPLACE

<<<<<<< SEARCH
                for pivot, coefficient in zip(
                    pivots, coefficients
                ):
                    full_qkv_weight[value_start + pivot] = (
                        full_qkv_weight[value_start + pivot]
                        + coefficient
                        * full_qkv_weight[value_start + target_row]
                    )

            block.attn.qkv.weight = nn.Parameter(
=======
                for pivot, coefficient in zip(
                    pivots, coefficients
                ):
                    full_qkv_weight[value_start + pivot] = (
                        full_qkv_weight[value_start + pivot]
                        + coefficient
                        * full_qkv_weight[value_start + target_row]
                    )

            # Use an unused scalar shear to zero a disjoint coordinate in
            # the preceding first-head row. Selecting the largest available
            # pivot avoids the three-by-three solve of the failed extension.
            aux_target = block.attn.head_dim - 2
            aux_column = relative_proj_weight.size(1) - 3
            aux_candidates = [
                row
                for row in range(block.attn.head_dim)
                if row != aux_target
            ]
            aux_pivot = max(
                aux_candidates,
                key=lambda row: abs(
                    relative_proj_weight[row, aux_column].item()
                ),
            )
            aux_coefficient = (
                relative_proj_weight[aux_target, aux_column]
                / relative_proj_weight[aux_pivot, aux_column]
            )
            relative_proj_weight[aux_target] = (
                relative_proj_weight[aux_target]
                - aux_coefficient
                * relative_proj_weight[aux_pivot]
            )
            full_qkv_weight[value_start + aux_pivot] = (
                full_qkv_weight[value_start + aux_pivot]
                + aux_coefficient
                * full_qkv_weight[value_start + aux_target]
            )

            block.attn.qkv.weight = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            first_target = block.attn.head_dim - 1
            block.attn.proj.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_proj_weight[:aux_target],
                        relative_proj_weight[
                            first_target + 1:-1
                        ],
                    ),
                    dim=0,
                )
            )
            block.attn.proj_aux_weight = nn.Parameter(
                torch.cat(
                    (
                        relative_proj_weight[
                            aux_target, :aux_column
                        ],
                        relative_proj_weight[
                            aux_target, aux_column + 1:
                        ],
                    )
                )
            )
            block.attn.proj_head_weight = nn.Parameter(
                relative_proj_weight[first_target, :-2]
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
=======
def reconstruct_attention_output_weight(
    weight, aux_weight, head_weight, last_weight, head_dim
):
    aux_relative = torch.cat(
        (
            aux_weight[:-2],
            aux_weight.new_zeros(1),
            aux_weight[-2:],
        )
    )
    head_relative = torch.cat(
        (head_weight, head_weight.new_zeros(2))
    )
    last_relative = torch.cat(
        (last_weight, last_weight.new_zeros(2))
    )
    split = head_dim - 2
    rows = torch.cat(
        (
            weight[:split],
            aux_relative.unsqueeze(0),
            head_relative.unsqueeze(0),
            weight[split:],
            last_relative.unsqueeze(0),
        ),
        dim=0,
    )
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
            proj_aux_weight,
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
                        proj_aux_weight.detach(),
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
            proj_aux_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
>>>>>>> REPLACE

<<<<<<< SEARCH
                reconstruct_attention_output_weight(
                    proj_weight,
                    proj_head_weight,
                    proj_last_weight,
                    head_dim,
                )
                @ omitted_value
=======
                reconstruct_attention_output_weight(
                    proj_weight,
                    proj_aux_weight,
                    proj_head_weight,
                    proj_last_weight,
                    head_dim,
                )
                @ omitted_value
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
        proj_aux_weight,
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
            proj_aux_weight.detach(),
            proj_head_weight.detach(),
            proj_last_weight.detach(),
            head_dim,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_head_weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
        )
=======
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_aux_weight,
            block.attn.proj_head_weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
        )
>>>>>>> REPLACE