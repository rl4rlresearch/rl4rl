MECHANISM: Max-pivot normalized value/output scale gauge

HYPOTHESIS: A scalar value/output gauge on the balanced 1501-parameter model will reduce it to 1500 parameters while retaining at least 99% accuracy, because normalizing one already-gauged projection row and absorbing its scale into the matching value row preserves the initialized function without imposing a third zero-coordinate constraint.

INTENDED_EDIT: Reproduce the verified final-scale, query-bias, embedding, and balanced two-coordinate projection quotients, then represent the first head’s gauged projection row by four max-pivot chart coordinates with fixed norm and transfer its omitted scale into the corresponding value weight.

EVIDENCE: The balanced two-coordinate-per-head design achieved 100% at 1501 parameters, while adding a third zero coordinate reached only 98.82% and its SVD variant reached 87.68%; this tests an unused diagonal value/output symmetry with a bounded max-pivot chart instead of another projection-direction restriction.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, and every value-bias coordinate can
        # be absorbed by the downstream projection bias. Store only query
        # bias and reconstruct the other two bias vectors in fixed gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
        # The feature-uniform component of this residual bias is canceled by
        # downstream LayerNorms, so retain only its relative coordinates.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. A query/key basis gauge fixes one query coordinate.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Downstream LayerNorms cancel the uniform output coordinate. Two
        # value/output shears are fixed in each head, and a remaining scalar
        # value/output gauge normalizes the first head's target row.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.register_buffer(
            "proj_head_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = self.qkv.bias[:d_model]
        value_bias = self.qkv.bias.new_zeros(d_model)
=======
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(1))
        )
        value_bias = self.qkv.bias.new_zeros(d_model)
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
        head_relative = torch.cat(
            (head_free, head_free.new_zeros(2))
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
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
=======
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        # One common positive affine scale changes only the global logit
        # temperature under protected argmax decoding.
        self.ln_f = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 1))
        self.ln_f_bias = nn.Parameter(torch.zeros(cfg.d_model))

        # Weight tying with input embeddings.
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
        # relative representatives. Two conditioned value/output shears are
        # fixed per head, then a diagonal gauge normalizes the first target
        # row through a max-pivot coordinate chart.
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
        x = self.ln_f(x)
        logits = F.linear(x, token_weight)
=======
        x = self.ln_f(x)
        ln_f_weight = torch.cat(
            (
                self.ln_f_weight,
                self.ln_f_weight.new_ones(1),
            )
        )
        x = x * ln_f_weight + self.ln_f_bias
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
def reconstruct_output_weight(param):
    return reconstruct_input_weight(param).transpose(0, 1)
=======
def reconstruct_attention_output_weight(
    weight,
    head_weight,
    last_weight,
    head_dim,
    head_pivot,
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
    head_relative = torch.cat(
        (head_free, head_free.new_zeros(2))
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
            head_pivot,
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
=======
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            head_pivot,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
            if grad is None:
                continue
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
                    head_pivot,
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
        term = omitted_grad.square().sum()
        total_sq = term if total_sq is None else total_sq + term
=======
    for (
        qkv_bias,
        proj_weight,
        proj_head_weight,
        proj_last_weight,
        proj_bias,
        head_dim,
        head_pivot,
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
        )
        omitted_grad = (
            full_proj_weight
            * full_proj_grad.unsqueeze(1)
        ).sum(dim=0)
        term = omitted_grad.square().sum()
        total_sq = term if total_sq is None else total_sq + term
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
            int(block.attn.proj_head_pivot.item()),
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE