MECHANISM: Balanced per-head value/output basis gauge fixing

HYPOTHESIS: Extending the second attention head from one to two function-preserving projection-coordinate gauges will reduce the model to 1501 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store five relative coordinates for each head’s final projection row, reconstruct two omitted coordinates per row, and initialize the second head with the same conditioned two-pivot basis change already used for the first head.

EVIDENCE: Reference Design 3 verified this balanced construction at 100% accuracy with 1501 parameters; both tested 1500-parameter extensions fell below 99%, making the qualified 1501 design the strongest supported reduction.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_relative = torch.cat(
            (
                self.proj_last_weight,
                self.proj_last_weight.new_zeros(1),
            )
        )
=======
        last_relative = torch.cat(
            (
                self.proj_last_weight,
                self.proj_last_weight.new_zeros(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.proj_last_weight = nn.Parameter(
                relative_proj_weight[-1, :-1]
            )
=======
            block.attn.proj_last_weight = nn.Parameter(
                relative_proj_weight[-1, :-2]
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
        (last_weight, last_weight.new_zeros(1))
    )
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
>>>>>>> REPLACE