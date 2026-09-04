MECHANISM: First-head affine query/key scale gauge

HYPOTHESIS: Normalizing one first-head query weight-and-bias coordinate and absorbing its scale into the matching key row will reduce the verified 1498-parameter model to 1497 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Add a max-pivot chart for the first head’s final affine query coordinate, remove its row and bias from the ordinary parameters, reconstruct them during inference, and transfer the omitted scale into the corresponding key row at initialization.

EVIDENCE: The current query/key scale gauge achieved 99.93% accuracy at 1498 parameters; extending the same exact diagonal symmetry to an independent head is better supported than another ordinary value/output-row normalization, which fell to 67.77%.

<<<<<<< SEARCH
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
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Independent diagonal query/key gauges normalize one
        # affine query coordinate per head.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
        self.q_secondary_weight = nn.Parameter(
            torch.empty(d_model - 1)
        )
        self.register_buffer(
            "q_secondary_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.register_buffer(
            "q_target_pivot",
            torch.zeros((), dtype=torch.long),
        )
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
        q_secondary = self.head_dim - 1
        q_secondary_pivot = int(self.q_secondary_pivot.item())
        q_secondary_chart = torch.cat(
            (
                self.q_secondary_weight[:q_secondary_pivot],
                self.q_secondary_weight.new_full((1,), 1.0),
                self.q_secondary_weight[q_secondary_pivot:],
            )
        )
        q_secondary_joint = q_secondary_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_secondary_chart.norm()
        )
        q_secondary_relative = q_secondary_joint[:-1]
        q_secondary_bias = q_secondary_joint[-1:]
        query_bias = torch.cat(
            (
                self.qkv.bias[:q_secondary],
                q_secondary_bias,
                self.qkv.bias[q_secondary:],
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
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_secondary],
                q_secondary_relative.unsqueeze(0),
                self.qkv.weight[q_secondary:],
            ),
            dim=0,
        )
        q_target_row = d_model - 1
        qkv_rows = torch.cat(
            (
                qkv_rows[:q_target_row],
                q_target_relative.unsqueeze(0),
                qkv_rows[q_target_row:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            key_target = cfg.d_model + q_target
            full_qkv_weight[key_target] = (
                q_target_scale * full_qkv_weight[key_target]
            )
            block.attn.q_target_pivot.fill_(q_target_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_secondary = block.attn.head_dim - 1
            q_secondary_joint = torch.cat(
                (
                    relative_qkv_weight[q_secondary],
                    relative_qkv_weight.new_zeros(1),
                )
            )
            q_secondary_pivot = int(
                q_secondary_joint.abs().argmax().item()
            )
            q_secondary_pivot_value = q_secondary_joint[
                q_secondary_pivot
            ]
            q_secondary_chart = (
                q_secondary_joint / q_secondary_pivot_value
            )
            q_secondary_gauge_norm = (
                0.02 * math.sqrt(cfg.d_model - 1)
            )
            q_secondary_scale = (
                q_secondary_pivot_value.sign()
                * q_secondary_joint.norm()
                / q_secondary_gauge_norm
            )
            key_secondary = cfg.d_model + q_secondary
            full_qkv_weight[key_secondary] = (
                q_secondary_scale
                * full_qkv_weight[key_secondary]
            )
            block.attn.q_secondary_pivot.fill_(
                q_secondary_pivot
            )

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_secondary],
                        relative_qkv_weight[
                            q_secondary + 1:q_target
                        ],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_secondary_weight = nn.Parameter(
                torch.cat(
                    (
                        q_secondary_chart[:q_secondary_pivot],
                        q_secondary_chart[q_secondary_pivot + 1:],
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
>>>>>>> REPLACE