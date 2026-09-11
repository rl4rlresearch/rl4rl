MECHANISM: Asymmetric third query-bias gauge with qualified second-head dual scale charts

HYPOTHESIS: Fixing the remaining trailing second-head query-bias coordinate while reproducing the qualified 1494-parameter dual query-row charts will yield 1493 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Reconstruct three trailing second-head query biases as zero and represent its final two query rows with independent max-pivot query/key scale charts.

EVIDENCE: The second-head dual-chart design reached 99.88% at 1494 parameters, while the analogous first-head penultimate chart collapsed to 74.49%; testing the unused third bias gauge in the already-successful second head is the closest distinct reduction.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final two query-bias
        # coordinates in each head; the final coordinate additionally uses
        # a diagonal query/key scale gauge.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.register_buffer(
            "q_target_pivot",
            torch.zeros((), dtype=torch.long),
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final two query-bias
        # coordinates in the first head and the final three in the second.
        # The final two second-head query rows additionally use diagonal
        # query/key scale gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 5))
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.register_buffer(
            "q_penultimate_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_target_pivot",
            torch.zeros((), dtype=torch.long),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = torch.cat(
            (
                self.qkv.bias[:bias_split],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[bias_split:],
                self.qkv.bias.new_zeros(2),
            )
        )
=======
        query_bias = torch.cat(
            (
                self.qkv.bias[:bias_split],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[bias_split:],
                self.qkv.bias.new_zeros(3),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
        q_penultimate_chart = torch.cat(
            (
                self.q_penultimate_weight[:q_penultimate_pivot],
                self.q_penultimate_weight.new_full((1,), 1.0),
                self.q_penultimate_weight[q_penultimate_pivot:],
            )
        )
        q_penultimate_relative = q_penultimate_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_penultimate_chart.norm()
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

        q_penultimate_row = d_model - 2
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_penultimate_row],
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_penultimate_row:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )

            q_penultimate = cfg.d_model - 2
            q_penultimate_free = relative_qkv_weight[q_penultimate]
            q_penultimate_pivot = int(
                q_penultimate_free.abs().argmax().item()
            )
            q_penultimate_pivot_value = q_penultimate_free[
                q_penultimate_pivot
            ]
            q_penultimate_chart = (
                q_penultimate_free / q_penultimate_pivot_value
            )
            q_penultimate_gauge_norm = (
                0.02 * math.sqrt(q_penultimate_free.numel())
            )
            q_penultimate_scale = (
                q_penultimate_pivot_value.sign()
                * q_penultimate_free.norm()
                / q_penultimate_gauge_norm
            )
            key_penultimate = cfg.d_model + q_penultimate
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale
                * full_qkv_weight[key_penultimate]
            )
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
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
                        relative_qkv_weight[:q_penultimate],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_penultimate_weight = nn.Parameter(
                torch.cat(
                    (
                        q_penultimate_chart[:q_penultimate_pivot],
                        q_penultimate_chart[
                            q_penultimate_pivot + 1:
                        ],
                    )
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