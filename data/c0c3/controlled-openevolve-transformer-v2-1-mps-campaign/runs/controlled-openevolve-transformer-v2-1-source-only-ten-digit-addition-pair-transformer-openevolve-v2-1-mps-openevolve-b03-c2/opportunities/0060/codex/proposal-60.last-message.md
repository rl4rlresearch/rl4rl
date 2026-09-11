MECHANISM: Max-pivot query/key coordinate scale gauge

HYPOTHESIS: Extending the qualified 1499-parameter design with a normalized chart for the zero-bias query coordinate will produce a 1498-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified balanced projection-row gauges, then normalize the final query-weight row and absorb its omitted scale into the matching key-weight row.

EVIDENCE: Independent normalized value/output gauges reached 99.35–99.64% at 1499 parameters, whereas additional zero-coordinate restrictions failed; the proposed reduction uses the analogous exact diagonal query/key symmetry on the query coordinate whose bias is already fixed to zero.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. A query/key basis gauge fixes the final query-bias
        # coordinate.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
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
        self.attn_drop = nn.Dropout(dropout)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The final zero-bias query coordinate additionally uses
        # a diagonal query/key scale gauge.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.register_buffer(
            "q_target_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.proj = nn.Linear(d_model, d_model)
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
        qkv_weight_relative = torch.cat(
            (
                self.qkv.weight,
                self.qkv.weight.new_zeros(
                    (self.qkv.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        qkv_weight = (
            qkv_weight_relative
            + self.qkv.weight.mean(dim=-1, keepdim=True)
        )
=======
        q_target_pivot = int(self.q_target_pivot.item())
        q_target_chart = torch.cat(
            (
                self.q_target_weight[:q_target_pivot],
                self.q_target_weight.new_full((1,), 1.0),
                self.q_target_weight[q_target_pivot:],
            )
        )
        q_target_relative = q_target_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_target_chart.norm()
        )
        q_target_row = d_model - 1
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_target_row],
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_target_row:],
            ),
            dim=0,
        )
        qkv_weight_relative = torch.cat(
            (
                qkv_rows,
                qkv_rows.new_zeros((qkv_rows.size(0), 1)),
            ),
            dim=-1,
        )
        qkv_weight = (
            qkv_weight_relative
            + qkv_rows.mean(dim=-1, keepdim=True)
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
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. Before fixing one projection coordinate,
        # apply an exact within-head value/output basis change so the fresh
        # initialization computes the same function.
        for block in self.blocks:
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
=======
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. Fix two conditioned value/output shears
        # per head and normalize both target rows. The final query coordinate
        # has zero bias, so its weight scale is transferred exactly into the
        # matching key coordinate.
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

            first_target = block.attn.head_dim - 1
            head_free = relative_proj_weight[first_target, :-2]
            head_pivot = int(head_free.abs().argmax().item())
            head_pivot_value = head_free[head_pivot]
            head_chart = head_free / head_pivot_value
            head_gauge_norm = 0.02 * math.sqrt(head_free.numel())
            head_scale = (
                head_pivot_value.sign()
                * head_free.norm()
                / head_gauge_norm
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

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_target = cfg.d_model - 1
            q_target_free = relative_qkv_weight[q_target]
            q_target_pivot = int(
                q_target_free.abs().argmax().item()
            )
            q_target_pivot_value = q_target_free[q_target_pivot]
            q_target_chart = (
                q_target_free / q_target_pivot_value
            )
            q_target_gauge_norm = (
                0.02 * math.sqrt(q_target_free.numel())
            )
            q_target_scale = (
                q_target_pivot_value.sign()
                * q_target_free.norm()
                / q_target_gauge_norm
            )
            key_target = cfg.d_model + q_target
            full_qkv_weight[key_target] = (
                q_target_scale * full_qkv_weight[key_target]
            )
            block.attn.q_target_pivot.fill_(q_target_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_target],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_target_weight = nn.Parameter(
                torch.cat(
                    (
                        q_target_chart[:q_target_pivot],
                        q_target_chart[q_target_pivot + 1:],
                    )
                )
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
        (head_weight, head_weight.new_zeros(1))
    )
    last_relative = torch.cat(
        (last_weight, last_weight.new_zeros(1))
    )
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                        proj_last_weight.detach(),
                        head_dim,
                    )
=======
                        proj_last_weight.detach(),
                        head_dim,
                        head_pivot,
                        last_pivot,
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
                    proj_last_weight,
                    head_dim,
                )
                @ omitted_value
=======
                    proj_last_weight,
                    head_dim,
                    head_pivot,
                    last_pivot,
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            proj_last_weight.detach(),
            head_dim,
        )
        omitted_grad = (
=======
            proj_last_weight.detach(),
            head_dim,
            head_pivot,
            last_pivot,
        )
        omitted_grad = (
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
        )
=======
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
            int(block.attn.proj_head_pivot.item()),
            int(block.attn.proj_last_pivot.item()),
        )
>>>>>>> REPLACE