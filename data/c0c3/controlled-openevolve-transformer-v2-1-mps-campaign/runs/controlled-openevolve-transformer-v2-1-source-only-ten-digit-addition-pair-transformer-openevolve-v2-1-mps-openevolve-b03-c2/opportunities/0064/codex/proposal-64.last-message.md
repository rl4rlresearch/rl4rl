MECHANISM: Dual zero-bias query/key scale charts

HYPOTHESIS: Applying the verified zero-bias query/key scale gauge to the first head as well as the second will reduce the qualified 1497-parameter model to 1496 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified per-head query-bias and normalized value/output gauges, then normalize both heads’ final zero-bias query rows and absorb their omitted scales into the matching key rows.

EVIDENCE: Reference Design 1 achieved 99.93% accuracy at 1497 parameters with both final query biases fixed and the second-head zero-bias query chart; the failed 1497 affine chart normalized a bias-bearing first-head coordinate, so removing that bias coupling makes the independent first-head scale gauge the most direct next test.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Within each head, an invertible query/key basis change
        # can additionally fix one generic query-bias coordinate to zero.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
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
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix each head's final query
        # bias, and diagonal gauges normalize both resulting zero-bias rows.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
        self.q_head_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_last_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.register_buffer(
            "q_head_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_last_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.proj = nn.Linear(d_model, d_model)
        # Two value/output shears are fixed in each head, and independent
        # scalar gauges normalize both heads' target projection rows.
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(1))
        )
        value_bias = self.qkv.bias.new_zeros(d_model)
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )
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
        bsz, seqlen, d_model = x.shape
        bias_split = self.head_dim - 1
        query_bias = torch.cat(
            (
                self.qkv.bias[:bias_split],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[bias_split:],
                self.qkv.bias.new_zeros(1),
            )
        )
        value_bias = self.qkv.bias.new_zeros(d_model)
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )

        q_head_pivot = int(self.q_head_pivot.item())
        q_head_chart = torch.cat(
            (
                self.q_head_weight[:q_head_pivot],
                self.q_head_weight.new_full((1,), 1.0),
                self.q_head_weight[q_head_pivot:],
            )
        )
        q_head_relative = q_head_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_head_chart.norm()
        )
        q_last_pivot = int(self.q_last_pivot.item())
        q_last_chart = torch.cat(
            (
                self.q_last_weight[:q_last_pivot],
                self.q_last_weight.new_full((1,), 1.0),
                self.q_last_weight[q_last_pivot:],
            )
        )
        q_last_relative = q_last_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_last_chart.norm()
        )

        q_head_row = self.head_dim - 1
        q_last_row = d_model - 1
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_head_row],
                q_head_relative.unsqueeze(0),
                self.qkv.weight[q_head_row:q_last_row - 1],
                q_last_relative.unsqueeze(0),
                self.qkv.weight[q_last_row - 1:],
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
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
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
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
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
        # relative representatives. In each head, use two well-conditioned
        # pivot directions to fix two projection coordinates while applying
        # the inverse value-weight basis change.
=======
        # Store LayerNorm-null directions through relative representatives.
        # Fix two conditioned value/output shears per head, normalize both
        # projection targets, and normalize both zero-bias query targets.
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
            q_head_target = block.attn.head_dim - 1
            q_head_free = relative_qkv_weight[q_head_target]
            q_head_pivot = int(
                q_head_free.abs().argmax().item()
            )
            q_head_pivot_value = q_head_free[q_head_pivot]
            q_head_chart = q_head_free / q_head_pivot_value
            q_head_gauge_norm = (
                0.02 * math.sqrt(q_head_free.numel())
            )
            q_head_scale = (
                q_head_pivot_value.sign()
                * q_head_free.norm()
                / q_head_gauge_norm
            )
            full_qkv_weight[cfg.d_model + q_head_target] = (
                q_head_scale
                * full_qkv_weight[cfg.d_model + q_head_target]
            )
            block.attn.q_head_pivot.fill_(q_head_pivot)

            q_last_target = cfg.d_model - 1
            q_last_free = relative_qkv_weight[q_last_target]
            q_last_pivot = int(
                q_last_free.abs().argmax().item()
            )
            q_last_pivot_value = q_last_free[q_last_pivot]
            q_last_chart = q_last_free / q_last_pivot_value
            q_last_gauge_norm = (
                0.02 * math.sqrt(q_last_free.numel())
            )
            q_last_scale = (
                q_last_pivot_value.sign()
                * q_last_free.norm()
                / q_last_gauge_norm
            )
            full_qkv_weight[cfg.d_model + q_last_target] = (
                q_last_scale
                * full_qkv_weight[cfg.d_model + q_last_target]
            )
            block.attn.q_last_pivot.fill_(q_last_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_head_target],
                        relative_qkv_weight[
                            q_head_target + 1:q_last_target
                        ],
                        relative_qkv_weight[q_last_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_head_weight = nn.Parameter(
                torch.cat(
                    (
                        q_head_chart[:q_head_pivot],
                        q_head_chart[q_head_pivot + 1:],
                    )
                )
            )
            block.attn.q_last_weight = nn.Parameter(
                torch.cat(
                    (
                        q_last_chart[:q_last_pivot],
                        q_last_chart[q_last_pivot + 1:],
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
        (head_weight, head_weight.new_zeros(2))
    )
    last_relative = torch.cat(
        (last_weight, last_weight.new_zeros(2))
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
                    proj_head_weight,
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