MECHANISM: Dual-target bias-preserving query/key shear

HYPOTHESIS: Shearing the already-successful second-head bias-bearing query row against both normalized zero-bias rows will reduce the verified 1491-parameter model to 1490 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified two-coordinate scale/shear chart, then omit both target-pivot coordinates from the same neighboring bias-bearing query row and absorb the inverse shears into the corresponding key rows.

EVIDENCE: The first bias-preserving shear on this row achieved 99.95% at 1491 parameters, whereas the failed 1490 design changed the other bias-bearing row and reached 71.83%; using the unused zero-bias target with the already-successful row isolates a distinct exact gauge.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final two query-bias
        # coordinates in each head. Both zero-bias trailing coordinates of
        # the second head additionally use diagonal query/key scale gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The second head's zero-bias query rows use their full
        # scale/shear gauge, while the neighboring bias-bearing row is
        # sheared against both of them without changing its learned bias.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
        q_target_pivot = int(self.q_target_pivot.item())
        query_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate
            not in (q_penultimate_pivot, q_target_pivot)
        ]

        q_penultimate_chart = self.q_penultimate_weight.new_zeros(
            d_model - 1
        )
        q_penultimate_chart[q_penultimate_pivot] = 1.0
        q_penultimate_chart[query_free] = (
            self.q_penultimate_weight
        )
        q_penultimate_relative = q_penultimate_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_penultimate_chart.norm()
        )

        q_target_chart = self.q_target_weight.new_zeros(
            d_model - 1
        )
        q_target_chart[q_target_pivot] = 1.0
        q_target_chart[query_free] = self.q_target_weight
        q_target_relative = q_target_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_target_chart.norm()
        )

        q_shear_relative = self.q_shear_weight.new_zeros(
            d_model - 1
        )
        q_shear_relative[query_free] = self.q_shear_weight
        q_shear_row = d_model - 3
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_shear_row],
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_shear_row:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            q_target_chart = q_target_free / q_target_pivot_value
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
=======
            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
            q_shear_free = relative_qkv_weight[q_shear]
            q_penultimate_free = relative_qkv_weight[q_penultimate]
            q_target_free = relative_qkv_weight[q_target]

            q_target_pivot = int(
                q_target_free.abs().argmax().item()
            )
            q_target_pivot_value = q_target_free[q_target_pivot]
            first_shear = (
                q_penultimate_free[q_target_pivot]
                / q_target_pivot_value
            )
            q_penultimate_sheared = (
                q_penultimate_free
                - first_shear * q_target_free
            )
            q_penultimate_sheared[q_target_pivot] = 0.0

            q_penultimate_pivot = int(
                q_penultimate_sheared.abs().argmax().item()
            )
            q_penultimate_pivot_value = q_penultimate_sheared[
                q_penultimate_pivot
            ]
            second_shear = (
                q_target_free[q_penultimate_pivot]
                / q_penultimate_pivot_value
            )
            q_target_sheared = (
                q_target_free
                - second_shear * q_penultimate_sheared
            )
            q_target_sheared[q_penultimate_pivot] = 0.0

            q_penultimate_chart = (
                q_penultimate_sheared
                / q_penultimate_pivot_value
            )
            q_penultimate_gauge_norm = (
                0.02 * math.sqrt(q_penultimate_sheared.numel())
            )
            q_penultimate_scale = (
                q_penultimate_pivot_value.sign()
                * q_penultimate_sheared.norm()
                / q_penultimate_gauge_norm
            )
            q_penultimate_relative = q_penultimate_chart * (
                q_penultimate_gauge_norm
                / q_penultimate_chart.norm()
            )

            q_target_pivot_value = q_target_sheared[q_target_pivot]
            q_target_chart = (
                q_target_sheared / q_target_pivot_value
            )
            q_target_gauge_norm = (
                0.02 * math.sqrt(q_target_sheared.numel())
            )
            q_target_scale = (
                q_target_pivot_value.sign()
                * q_target_sheared.norm()
                / q_target_gauge_norm
            )
            q_target_relative = q_target_chart * (
                q_target_gauge_norm / q_target_chart.norm()
            )

            bias_target_shear = (
                q_shear_free[q_target_pivot]
                / q_target_relative[q_target_pivot]
            )
            q_shear_chart = (
                q_shear_free
                - bias_target_shear * q_target_relative
            )
            q_shear_chart[q_target_pivot] = 0.0
            bias_penultimate_shear = (
                q_shear_chart[q_penultimate_pivot]
                / q_penultimate_relative[q_penultimate_pivot]
            )
            q_shear_chart = (
                q_shear_chart
                - bias_penultimate_shear
                * q_penultimate_relative
            )
            q_shear_chart[q_penultimate_pivot] = 0.0
            q_shear_chart[q_target_pivot] = 0.0

            key_shear = cfg.d_model + q_shear
            key_penultimate = cfg.d_model + q_penultimate
            key_target = cfg.d_model + q_target
            key_penultimate_free = full_qkv_weight[
                key_penultimate
            ].clone()
            key_target_free = full_qkv_weight[key_target].clone()
            key_target_sheared = (
                key_target_free
                + first_shear * key_penultimate_free
            )
            key_penultimate_sheared = (
                key_penultimate_free
                + second_shear * key_target_sheared
            )
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale * key_penultimate_sheared
                + bias_penultimate_shear
                * full_qkv_weight[key_shear]
            )
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + bias_target_shear
                * full_qkv_weight[key_shear]
            )
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
            )
            block.attn.q_target_pivot.fill_(q_target_pivot)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_shear],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            query_free = [
                coordinate
                for coordinate in range(q_target_chart.numel())
                if coordinate
                not in (q_penultimate_pivot, q_target_pivot)
            ]
            block.attn.q_shear_weight = nn.Parameter(
                q_shear_chart[query_free].clone()
            )
            block.attn.q_penultimate_weight = nn.Parameter(
                q_penultimate_chart[query_free].clone()
            )
            block.attn.q_target_weight = nn.Parameter(
                q_target_chart[query_free].clone()
            )
>>>>>>> REPLACE