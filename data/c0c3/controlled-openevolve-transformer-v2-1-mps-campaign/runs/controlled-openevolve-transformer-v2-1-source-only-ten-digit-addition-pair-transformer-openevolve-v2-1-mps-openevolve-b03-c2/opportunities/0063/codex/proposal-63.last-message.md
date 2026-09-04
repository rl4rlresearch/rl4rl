MECHANISM: Independent per-head query-bias gauge plus qualified zero-bias query/key scale chart

HYPOTHESIS: Fixing the final query-bias coordinate in each head while applying the verified scale chart to the second head’s zero-bias query row will produce a 1497-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified 1498-parameter query/key chart, additionally omit the first head’s final query-bias coordinate, and reconstruct both per-head omitted biases as zero.

EVIDENCE: The zero-bias query/key chart reached 99.90–99.93% at 1498 parameters, while the 1497 affine chart on a bias-bearing first-head coordinate reached only 42.83%; isolating the already-successful query-bias basis gauge in the first head tests whether the affine bias coupling caused that failure.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. A query/key basis gauge fixes the final query-bias
        # coordinate.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Independent query/key basis gauges fix each head's
        # final query-bias coordinate; the second zero-bias query coordinate
        # additionally uses a diagonal query/key scale gauge.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.register_buffer(
            "q_target_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            block.attn.qkv.weight = nn.Parameter(
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.proj.weight = nn.Parameter(
=======
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
>>>>>>> REPLACE