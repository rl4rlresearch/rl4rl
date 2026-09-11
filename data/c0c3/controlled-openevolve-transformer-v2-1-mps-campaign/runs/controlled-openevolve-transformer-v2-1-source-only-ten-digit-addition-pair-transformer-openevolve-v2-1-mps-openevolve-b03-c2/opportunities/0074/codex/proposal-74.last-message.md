MECHANISM: Second bias-preserving query/key shear

HYPOTHESIS: Extending the verified 1491-parameter construction with an exact shear from the remaining bias-bearing second-head query row into its normalized zero-bias target row will yield 1490 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified two-coordinate scale/shear chart and neighboring bias-preserving shear, then omit the target-pivot coordinate from the other bias-bearing query row and absorb its inverse shear into the target key row.

EVIDENCE: Reference Design 2 reached 99.95% accuracy at 1491 parameters after the first bias-preserving shear; the remaining second-head bias-bearing row admits the same transformation without imposing the failed additional query-bias restriction.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The final two query-bias coordinates in each head are
        # fixed. The second head's trailing query rows use two scale gauges
        # and an off-diagonal query/key shear gauge.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The final two query-bias coordinates in each head are
        # fixed. The second head's zero-bias pair uses its full scale/shear
        # gauge, while both bias-bearing rows are sheared against the target.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_outer_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 2)
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
        q_target_pivot = int(self.q_target_pivot.item())
        q_penultimate_chart = self.q_penultimate_weight.new_zeros(
            d_model - 1
        )
        q_penultimate_chart[q_penultimate_pivot] = 1.0
        q_penultimate_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate
            not in (q_penultimate_pivot, q_target_pivot)
        ]
        q_penultimate_chart[q_penultimate_free] = (
            self.q_penultimate_weight
        )
        q_penultimate_relative = q_penultimate_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_penultimate_chart.norm()
        )

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

        q_outer_relative = torch.cat(
            (
                self.q_outer_weight[:q_target_pivot],
                self.q_outer_weight.new_zeros(1),
                self.q_outer_weight[q_target_pivot:],
            )
        )
        q_shear_relative = torch.cat(
            (
                self.q_shear_weight[:q_target_pivot],
                self.q_shear_weight.new_zeros(1),
                self.q_shear_weight[q_target_pivot:],
            )
        )
        q_outer_row = d_model - 4
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_outer_row],
                q_outer_relative.unsqueeze(0),
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_outer_row:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
            q_penultimate_free = relative_qkv_weight[q_penultimate]
            q_target_free = relative_qkv_weight[q_target]

            q_target_pivot = int(
                q_target_free.abs().argmax().item()
            )
            q_target_pivot_value = q_target_free[q_target_pivot]
            shear_coefficient = (
                q_penultimate_free[q_target_pivot]
                / q_target_pivot_value
            )
            q_penultimate_sheared = (
                q_penultimate_free
                - shear_coefficient * q_target_free
            )
            q_penultimate_sheared[q_target_pivot] = 0.0

            q_penultimate_pivot = int(
                q_penultimate_sheared.abs().argmax().item()
            )
            q_penultimate_pivot_value = q_penultimate_sheared[
                q_penultimate_pivot
            ]
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

            key_penultimate = cfg.d_model + q_penultimate
            key_target = cfg.d_model + q_target
            full_qkv_weight[key_target] = q_target_scale * (
                full_qkv_weight[key_target]
                + shear_coefficient
                * full_qkv_weight[key_penultimate]
            )
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale
                * full_qkv_weight[key_penultimate]
            )
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
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
            q_penultimate_indices = [
                coordinate
                for coordinate in range(q_penultimate_chart.numel())
                if coordinate
                not in (q_penultimate_pivot, q_target_pivot)
            ]
            block.attn.q_penultimate_weight = nn.Parameter(
                q_penultimate_chart[q_penultimate_indices].clone()
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
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_outer = cfg.d_model - 4
            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
            q_outer_free = relative_qkv_weight[q_outer]
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

            q_target_pivot_value = q_target_sheared[
                q_target_pivot
            ]
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

            q_outer_coefficient = (
                q_outer_free[q_target_pivot]
                / q_target_relative[q_target_pivot]
            )
            q_outer_chart = (
                q_outer_free
                - q_outer_coefficient * q_target_relative
            )
            q_outer_chart[q_target_pivot] = 0.0

            q_shear_coefficient = (
                q_shear_free[q_target_pivot]
                / q_target_relative[q_target_pivot]
            )
            q_shear_chart = (
                q_shear_free
                - q_shear_coefficient * q_target_relative
            )
            q_shear_chart[q_target_pivot] = 0.0

            key_outer = cfg.d_model + q_outer
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
            )
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + q_outer_coefficient * full_qkv_weight[key_outer]
                + q_shear_coefficient * full_qkv_weight[key_shear]
            )
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
            )
            block.attn.q_target_pivot.fill_(q_target_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_outer],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_outer_weight = nn.Parameter(
                torch.cat(
                    (
                        q_outer_chart[:q_target_pivot],
                        q_outer_chart[q_target_pivot + 1:],
                    )
                )
            )
            block.attn.q_shear_weight = nn.Parameter(
                torch.cat(
                    (
                        q_shear_chart[:q_target_pivot],
                        q_shear_chart[q_target_pivot + 1:],
                    )
                )
            )
            query_free = [
                coordinate
                for coordinate in range(q_target_chart.numel())
                if coordinate
                not in (q_penultimate_pivot, q_target_pivot)
            ]
            block.attn.q_penultimate_weight = nn.Parameter(
                q_penultimate_chart[query_free].clone()
            )
            block.attn.q_target_weight = nn.Parameter(
                q_target_chart[query_free].clone()
            )
>>>>>>> REPLACE