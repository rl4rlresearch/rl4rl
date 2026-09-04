MECHANISM: Conditioned third-coordinate value/output basis gauge

HYPOTHESIS: Extending the verified balanced two-coordinate gauges by fixing a third relative projection coordinate in one attention head will reduce the model from 1501 to 1500 parameters while retaining at least 99% accuracy, because a best-conditioned three-pivot within-head basis change preserves the freshly initialized function.

INTENDED_EDIT: Reproduce the qualified query-bias, final-LayerNorm scale, embedding, and balanced projection quotients; then store four rather than five coordinates for the first head’s gauged projection row, with initialization selecting the best-conditioned omitted coordinate triple.

EVIDENCE: Reference Design 3 achieved 100% accuracy at 1501 parameters after fixing two projection coordinates in each head. The proposed one-parameter reduction uses the same successful value/output symmetry, while adaptive pivot-coordinate selection addresses the conditioning risk of extending the gauge to three coordinates.

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
        # Downstream LayerNorms cancel the uniform output coordinate.
        # Value/output basis gauges fix three relative coordinates in the
        # first head's target row and two in the second head's target row.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.register_buffer(
            "proj_head_keep",
            torch.arange(d_model - 4),
        )
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = self.qkv.bias[:d_model]
=======
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(1))
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
        head_relative = self.proj_head_weight.new_zeros(
            self.proj.weight.size(1)
        ).scatter(
            0, self.proj_head_keep, self.proj_head_weight
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
                weight_rows.new_zeros(
                    (weight_rows.size(0), 1)
                ),
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
        # Fix one common positive affine scale, which changes only the global
        # logit temperature under protected argmax decoding.
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
=======
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. Use a conditioned three-coordinate
        # value/output gauge in the first head and the qualified
        # two-coordinate gauge in the second.
        for block in self.blocks:
            full_qkv_weight = block.attn.qkv.weight.detach().clone()
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1).clone()
            )
            relative_proj_weight = (
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )
            value_start = 2 * cfg.d_model

            first_target = block.attn.head_dim - 1
            first_pivots = tuple(range(first_target))
            coordinate_triples = [
                (left, middle, right)
                for left in range(relative_proj_weight.size(1))
                for middle in range(
                    left + 1, relative_proj_weight.size(1)
                )
                for right in range(
                    middle + 1, relative_proj_weight.size(1)
                )
            ]
            first_omitted = max(
                coordinate_triples,
                key=lambda triple: abs(
                    torch.linalg.det(
                        relative_proj_weight[
                            list(first_pivots)
                        ][:, list(triple)]
                    ).item()
                ),
            )
            first_matrix = relative_proj_weight[
                list(first_pivots)
            ][:, list(first_omitted)].transpose(0, 1)
            first_coefficients = torch.linalg.solve(
                first_matrix,
                relative_proj_weight[
                    first_target, list(first_omitted)
                ],
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

            second_target = cfg.d_model - 1
            second_start = cfg.d_model - block.attn.head_dim
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
            first_keep = torch.tensor(
                [
                    coordinate
                    for coordinate in range(
                        relative_proj_weight.size(1)
                    )
                    if coordinate not in first_omitted
                ],
                dtype=torch.long,
                device=relative_proj_weight.device,
            )
            block.attn.proj_head_keep.copy_(first_keep)
            block.attn.proj_head_weight = nn.Parameter(
                relative_proj_weight[first_target, first_keep]
            )
            block.attn.proj_last_weight = nn.Parameter(
                relative_proj_weight[-1, :-2]
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
    weight, head_weight, head_keep, last_weight, head_dim
):
    head_relative = head_weight.new_zeros(
        weight.size(1)
    ).scatter(0, head_keep, head_weight)
    last_relative = torch.cat(
        (
            last_weight,
            last_weight.new_zeros(
                weight.size(1) - last_weight.numel()
            ),
        )
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
            proj_head_keep,
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
                        proj_head_keep,
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
=======
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_head_keep,
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
                    proj_head_keep,
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
        proj_head_keep,
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
            proj_head_keep,
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
            block.attn.proj_head_keep,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE