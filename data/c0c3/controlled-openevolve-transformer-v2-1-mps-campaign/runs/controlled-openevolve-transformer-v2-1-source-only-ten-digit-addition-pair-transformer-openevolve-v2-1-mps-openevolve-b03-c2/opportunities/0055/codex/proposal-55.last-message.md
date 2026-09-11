MECHANISM: Orthogonal nullspace value/output basis gauge

HYPOTHESIS: Replacing the failed three-pivot shear with a norm-preserving SVD basis rotation will reduce the balanced 1501-parameter design to 1500 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Fix three relative projection coordinates in the first attention head and two in the second; use an orthogonal first-head initialization transform and the verified conditioned two-pivot transform for the second head.

EVIDENCE: The balanced two-coordinate-per-head design reached 100% at 1501 parameters, while the three-coordinate pivot construction narrowly missed at 98.82%; an orthogonal transform tests whether conditioning, rather than the additional gauge itself, caused that loss.

<<<<<<< SEARCH
        # Downstream LayerNorms cancel the feature-uniform output coordinate.
        # Independent value/output basis gauges fix one relative weight in
        # the final projection row of each attention head.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
=======
        # Downstream LayerNorms cancel the feature-uniform output coordinate.
        # Value/output basis gauges fix three relative coordinates in the
        # first head's final row and two in the second head's final row.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_relative = torch.cat(
            (
                self.proj_head_weight,
                self.proj_head_weight.new_zeros(1),
            )
        )
        last_relative = torch.cat(
            (
                self.proj_last_weight,
                self.proj_last_weight.new_zeros(1),
            )
        )
=======
        head_relative = torch.cat(
            (
                self.proj_head_weight,
                self.proj_head_weight.new_zeros(3),
            )
        )
        last_relative = torch.cat(
            (
                self.proj_last_weight,
                self.proj_last_weight.new_zeros(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. Before fixing one projection coordinate,
        # apply an exact within-head value/output basis change so the fresh
        # initialization computes the same function.
=======
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. Use a norm-preserving nullspace rotation
        # for the three-coordinate first-head gauge and a conditioned
        # two-pivot change for the second head.
>>>>>>> REPLACE

<<<<<<< SEARCH
            target_col = cfg.d_model - 2
            target_rows = (
                block.attn.head_dim - 1,
                cfg.d_model - 1,
            )
            value_start = 2 * cfg.d_model
            for target_row in target_rows:
                head_start = (
                    target_row - block.attn.head_dim + 1
                )
                pivot_row = head_start + int(
                    relative_proj_weight[
                        head_start:target_row, target_col
                    ].abs().argmax().item()
                )
                coefficient = (
                    relative_proj_weight[target_row, target_col]
                    / relative_proj_weight[pivot_row, target_col]
                )
                relative_proj_weight[target_row] = (
                    relative_proj_weight[target_row]
                    - coefficient * relative_proj_weight[pivot_row]
                )
                full_qkv_weight[value_start + pivot_row] = (
                    full_qkv_weight[value_start + pivot_row]
                    + coefficient
                    * full_qkv_weight[value_start + target_row]
                )
=======
            value_start = 2 * cfg.d_model
            first_target = block.attn.head_dim - 1

            # The final right-singular vector is orthogonal to the three
            # selected projection columns. Using the complete orthogonal
            # basis avoids the amplification of a three-pivot shear.
            _, _, first_transform = torch.linalg.svd(
                relative_proj_weight[
                    :block.attn.head_dim, -3:
                ].transpose(0, 1),
                full_matrices=True,
            )
            relative_proj_weight[:block.attn.head_dim] = (
                first_transform
                @ relative_proj_weight[:block.attn.head_dim]
            )
            full_qkv_weight[
                value_start:value_start + block.attn.head_dim
            ] = (
                first_transform
                @ full_qkv_weight[
                    value_start:value_start + block.attn.head_dim
                ]
            )
            relative_proj_weight[first_target, -3:] = 0.0

            # Retain the verified conditioned two-coordinate construction
            # independently in the second head.
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.proj_head_weight = nn.Parameter(
                relative_proj_weight[first_target, :-1]
            )
            block.attn.proj_last_weight = nn.Parameter(
                relative_proj_weight[-1, :-1]
            )
=======
            block.attn.proj_head_weight = nn.Parameter(
                relative_proj_weight[first_target, :-3]
            )
            block.attn.proj_last_weight = nn.Parameter(
                relative_proj_weight[-1, :-2]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    head_relative = torch.cat(
        (head_weight, head_weight.new_zeros(1))
    )
    last_relative = torch.cat(
        (last_weight, last_weight.new_zeros(1))
    )
=======
    head_relative = torch.cat(
        (
            head_weight,
            head_weight.new_zeros(
                weight.size(1) - head_weight.numel()
            ),
        )
    )
    last_relative = torch.cat(
        (
            last_weight,
            last_weight.new_zeros(
                weight.size(1) - last_weight.numel()
            ),
        )
    )
>>>>>>> REPLACE