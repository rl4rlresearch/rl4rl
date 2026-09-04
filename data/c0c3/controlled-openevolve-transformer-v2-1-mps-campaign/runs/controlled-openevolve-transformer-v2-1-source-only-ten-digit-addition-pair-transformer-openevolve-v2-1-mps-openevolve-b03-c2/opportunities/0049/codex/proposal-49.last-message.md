MECHANISM: Per-head attention value/output basis gauge fixing

HYPOTHESIS: Fixing the same relative attention-output coordinate independently in the first head will reduce the verified 1504-parameter model to 1503 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Store six relative coordinates for one projection row in each attention head, reconstruct both fixed coordinates as zero, and apply function-preserving within-head value/output basis changes at initialization.

EVIDENCE: The current 1504-parameter design achieved 99.79% after one value/output gauge; using the independent symmetry of the other head extends that successful mechanism without adding another query-bias restriction, whose tested 1504-parameter variants failed.

<<<<<<< SEARCH
        # Downstream LayerNorms cancel the feature-uniform output coordinate.
        # A value/output basis gauge additionally fixes one relative weight.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_relative = torch.cat(
            (
                self.proj_last_weight,
                self.proj_last_weight.new_zeros(1),
            )
        )
        weight_rows = torch.cat(
            (self.proj.weight, last_relative.unsqueeze(0)),
            dim=0,
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_qkv_weight = block.attn.qkv.weight.detach().clone()
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1).clone()
            )
            relative_proj_weight = (
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )

            target_row = cfg.d_model - 1
            target_col = cfg.d_model - 2
            head_start = cfg.d_model - block.attn.head_dim
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
            value_start = 2 * cfg.d_model
            full_qkv_weight[value_start + pivot_row] = (
                full_qkv_weight[value_start + pivot_row]
                + coefficient
                * full_qkv_weight[value_start + target_row]
            )

            block.attn.qkv.weight = nn.Parameter(
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.proj.weight = nn.Parameter(
                relative_proj_weight[:-1]
            )
            block.attn.proj_last_weight = nn.Parameter(
                relative_proj_weight[-1, :-1]
            )
=======
            full_qkv_weight = block.attn.qkv.weight.detach().clone()
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1).clone()
            )
            relative_proj_weight = (
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )

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
                relative_proj_weight[first_target, :-1]
            )
            block.attn.proj_last_weight = nn.Parameter(
                relative_proj_weight[-1, :-1]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
def reconstruct_attention_output_weight(weight, last_weight):
    last_relative = torch.cat(
        (last_weight, last_weight.new_zeros(1))
    )
    rows = torch.cat(
        (weight, last_relative.unsqueeze(0)),
        dim=0,
    )
=======
def reconstruct_attention_output_weight(
    weight, head_weight, last_weight, head_dim
):
    head_relative = torch.cat(
        (head_weight, head_weight.new_zeros(1))
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            qkv_bias,
            proj_weight,
            proj_last_weight,
            proj_bias,
        ) in self.value_bias_specs:
=======
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
        ) in self.value_bias_specs:
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_proj_weight = (
                    reconstruct_attention_output_weight(
                        proj_weight.detach(),
                        proj_last_weight.detach(),
                    )
                )
=======
                full_proj_weight = (
                    reconstruct_attention_output_weight(
                        proj_weight.detach(),
                        proj_head_weight.detach(),
                        proj_last_weight.detach(),
                        head_dim,
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            qkv_bias,
            proj_weight,
            proj_last_weight,
            proj_bias,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
=======
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
                reconstruct_attention_output_weight(
                    proj_weight, proj_last_weight
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
                )
                @ omitted_value
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for (
        qkv_bias,
        proj_weight,
        proj_last_weight,
        proj_bias,
    ) in value_bias_specs:
=======
    for (
        qkv_bias,
        proj_weight,
        proj_head_weight,
        proj_last_weight,
        proj_bias,
        head_dim,
    ) in value_bias_specs:
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_proj_weight = reconstruct_attention_output_weight(
            proj_weight.detach(), proj_last_weight.detach()
        )
=======
        full_proj_weight = reconstruct_attention_output_weight(
            proj_weight.detach(),
            proj_head_weight.detach(),
            proj_last_weight.detach(),
            head_dim,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
        )
=======
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_head_weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
        )
>>>>>>> REPLACE