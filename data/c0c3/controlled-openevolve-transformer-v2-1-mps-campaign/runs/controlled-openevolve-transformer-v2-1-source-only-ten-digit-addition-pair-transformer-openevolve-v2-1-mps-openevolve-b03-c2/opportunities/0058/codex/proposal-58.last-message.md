MECHANISM: Additional max-pivot value/output scale gauge

HYPOTHESIS: Normalizing one ordinary first-head projection row and absorbing its scale into the matching value-weight row will reduce the qualified 1499-parameter model to 1498 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Apply the verified normalized scalar gauges to both target rows, then use the same function-preserving chart for one additional first-head projection row and propagate its pivot through reconstruction and optimization.

EVIDENCE: Independent scalar gauges achieved 99.94% at 1500 parameters and 99.64% at 1499, while additional zero-coordinate constraints failed; extending the successful scale symmetry is therefore the most supported next reduction.

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
        # shears are fixed in each head; scalar value/output gauges normalize
        # both target rows and one additional row in the first head.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_first_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.register_buffer(
            "proj_first_pivot",
            torch.zeros((), dtype=torch.long),
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
        first_pivot = int(self.proj_first_pivot.item())
        first_chart = torch.cat(
            (
                self.proj_first_weight[:first_pivot],
                self.proj_first_weight.new_full((1,), 1.0),
                self.proj_first_weight[first_pivot:],
            )
        )
        first_relative = first_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / first_chart.norm()
        )
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
        split = self.head_dim - 2
        weight_rows = torch.cat(
            (
                first_relative.unsqueeze(0),
                self.proj.weight[:split],
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
        # Store LayerNorm-null directions through relative representatives.
        # Two conditioned shears are fixed per head, then diagonal gauges
        # normalize both target rows and one additional first-head row.
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            first_row = 0
            first_free = relative_proj_weight[first_row]
            first_pivot = int(first_free.abs().argmax().item())
            first_pivot_value = first_free[first_pivot]
            first_chart = first_free / first_pivot_value
            first_gauge_norm = 0.02 * math.sqrt(first_free.numel())
            first_scale = (
                first_pivot_value.sign()
                * first_free.norm()
                / first_gauge_norm
            )
            full_qkv_weight[value_start + first_row] = (
                first_scale
                * full_qkv_weight[value_start + first_row]
            )
            block.attn.proj_first_pivot.fill_(first_pivot)

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

            block.attn.qkv.weight = nn.Parameter(
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.proj.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_proj_weight[
                            first_row + 1:first_target
                        ],
                        relative_proj_weight[
                            first_target + 1:-1
                        ],
                    ),
                    dim=0,
                )
            )
            block.attn.proj_first_weight = nn.Parameter(
                torch.cat(
                    (
                        first_chart[:first_pivot],
                        first_chart[first_pivot + 1:],
                    )
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
    first_weight,
    head_weight,
    last_weight,
    head_dim,
    first_pivot,
    head_pivot,
    last_pivot,
):
    first_chart = torch.cat(
        (
            first_weight[:first_pivot],
            first_weight.new_full((1,), 1.0),
            first_weight[first_pivot:],
        )
    )
    first_relative = first_chart * (
        (0.02 * math.sqrt(first_weight.numel() + 1))
        / first_chart.norm()
    )
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
    split = head_dim - 2
    rows = torch.cat(
        (
            first_relative.unsqueeze(0),
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
            proj_first_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            first_pivot,
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
                        proj_first_weight.detach(),
                        proj_head_weight.detach(),
                        proj_last_weight.detach(),
                        head_dim,
                        first_pivot,
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
            proj_first_weight,
            proj_head_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            first_pivot,
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
                    proj_first_weight,
                    proj_head_weight,
                    proj_last_weight,
                    head_dim,
                    first_pivot,
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
=======
    for (
        qkv_bias,
        proj_weight,
        proj_first_weight,
        proj_head_weight,
        proj_last_weight,
        proj_bias,
        head_dim,
        first_pivot,
        head_pivot,
        last_pivot,
    ) in value_bias_specs:
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_proj_weight = reconstruct_attention_output_weight(
            proj_weight.detach(),
            proj_head_weight.detach(),
            proj_last_weight.detach(),
            head_dim,
        )
=======
        full_proj_weight = reconstruct_attention_output_weight(
            proj_weight.detach(),
            proj_first_weight.detach(),
            proj_head_weight.detach(),
            proj_last_weight.detach(),
            head_dim,
            first_pivot,
            head_pivot,
            last_pivot,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    value_bias_specs = [
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_first_weight,
            block.attn.proj_head_weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
            int(block.attn.proj_first_pivot.item()),
            int(block.attn.proj_head_pivot.item()),
            int(block.attn.proj_last_pivot.item()),
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE