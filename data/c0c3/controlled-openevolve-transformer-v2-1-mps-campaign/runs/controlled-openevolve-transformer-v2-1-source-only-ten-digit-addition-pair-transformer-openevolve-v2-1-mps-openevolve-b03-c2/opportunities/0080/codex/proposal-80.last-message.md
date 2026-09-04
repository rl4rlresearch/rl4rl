MECHANISM: Conditioned complementary mutual query/key shear

HYPOTHESIS: Extending the qualified 1489-parameter construction with the complementary shear between the same two freely biased second-head query rows will produce 1488 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce Reference Design 1, then eliminate one additional coordinate from its previously sheared bias-bearing query row using the other bias-bearing row, with compensating zero-bias-row components and the inverse key transformation.

EVIDENCE: Reference Design 1 achieved 99.99% at 1489 parameters using the first mutual bias-bearing shear, whereas reductions involving another bias restriction, the first head, an affine scale, or a third value/output shear failed; the complementary gauge stays within the successful freely biased pair.

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
        # output bias. The second head's zero-bias query rows use their full
        # scale/shear gauge. Its freely biased rows use two conditioned,
        # complementary mutual shears.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_anchor_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.register_buffer(
            "q_anchor_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_complement_pivot",
            torch.zeros((), dtype=torch.long),
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
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_complement_pivot = int(
            self.q_complement_pivot.item()
        )
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
        q_shear_free = [
            coordinate
            for coordinate in query_free
            if coordinate != q_complement_pivot
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
        q_shear_relative[q_shear_free] = self.q_shear_weight
        q_anchor_relative = torch.cat(
            (
                self.q_anchor_weight[:q_anchor_pivot],
                self.q_anchor_weight.new_zeros(1),
                self.q_anchor_weight[q_anchor_pivot:],
            )
        )

        q_anchor_row = d_model - 4
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_anchor_row],
                q_anchor_relative.unsqueeze(0),
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_anchor_row:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. Fix two conditioned value/output shears
        # per head and normalize both target rows. The final query coordinate
        # has zero bias, so its weight scale is transferred exactly into the
        # matching key coordinate.
=======
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. Fix two conditioned value/output shears
        # per head and normalize both target rows. In the second query head,
        # fix the zero-bias scale/shear gauge and two conditioned mutual
        # shears between its freely biased rows.
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
=======
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_anchor = cfg.d_model - 4
            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
            q_anchor_free = relative_qkv_weight[q_anchor]
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

            q_anchor_pivot = int(
                q_shear_chart.abs().argmax().item()
            )
            anchor_shear = (
                q_anchor_free[q_anchor_pivot]
                / q_shear_chart[q_anchor_pivot]
            )
            q_anchor_chart = (
                q_anchor_free - anchor_shear * q_shear_chart
            )
            q_anchor_chart[q_anchor_pivot] = 0.0

            complement_penultimate = (
                q_anchor_chart[q_penultimate_pivot]
                / q_penultimate_relative[q_penultimate_pivot]
            )
            complement_target = (
                q_anchor_chart[q_target_pivot]
                / q_target_relative[q_target_pivot]
            )
            q_anchor_residual = (
                q_anchor_chart
                - complement_penultimate * q_penultimate_relative
                - complement_target * q_target_relative
            )
            q_anchor_residual[q_penultimate_pivot] = 0.0
            q_anchor_residual[q_target_pivot] = 0.0
            complement_candidates = [
                coordinate
                for coordinate in range(q_anchor_residual.numel())
                if coordinate not in (
                    q_anchor_pivot,
                    q_penultimate_pivot,
                    q_target_pivot,
                )
            ]
            q_complement_pivot = max(
                complement_candidates,
                key=lambda coordinate: abs(
                    q_anchor_residual[coordinate].item()
                ),
            )
            complement_shear = (
                q_shear_chart[q_complement_pivot]
                / q_anchor_residual[q_complement_pivot]
            )
            q_shear_chart = (
                q_shear_chart
                - complement_shear * q_anchor_residual
            )
            q_shear_chart[q_complement_pivot] = 0.0
            q_shear_chart[q_penultimate_pivot] = 0.0
            q_shear_chart[q_target_pivot] = 0.0

            key_anchor = cfg.d_model + q_anchor
            key_shear = cfg.d_model + q_shear
            key_penultimate = cfg.d_model + q_penultimate
            key_target = cfg.d_model + q_target
            key_anchor_free = full_qkv_weight[key_anchor].clone()
            key_shear_free = full_qkv_weight[key_shear].clone()
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
            key_penultimate_chart = (
                q_penultimate_scale * key_penultimate_sheared
                + bias_penultimate_shear * key_shear_free
            )
            key_target_chart = (
                q_target_scale * key_target_sheared
                + bias_target_shear * key_shear_free
            )
            key_shear_sheared = (
                key_shear_free + anchor_shear * key_anchor_free
            )
            full_qkv_weight[key_anchor] = (
                key_anchor_free
                + complement_shear * key_shear_sheared
            )
            full_qkv_weight[key_shear] = key_shear_sheared
            full_qkv_weight[key_penultimate] = (
                key_penultimate_chart
                - complement_shear
                * complement_penultimate
                * key_shear_sheared
            )
            full_qkv_weight[key_target] = (
                key_target_chart
                - complement_shear
                * complement_target
                * key_shear_sheared
            )
            block.attn.q_anchor_pivot.fill_(q_anchor_pivot)
            block.attn.q_complement_pivot.fill_(
                q_complement_pivot
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
                        relative_qkv_weight[:q_anchor],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_anchor_weight = nn.Parameter(
                torch.cat(
                    (
                        q_anchor_chart[:q_anchor_pivot],
                        q_anchor_chart[q_anchor_pivot + 1:],
                    )
                )
            )
            query_free = [
                coordinate
                for coordinate in range(q_target_chart.numel())
                if coordinate
                not in (q_penultimate_pivot, q_target_pivot)
            ]
            q_shear_free = [
                coordinate
                for coordinate in query_free
                if coordinate != q_complement_pivot
            ]
            block.attn.q_shear_weight = nn.Parameter(
                q_shear_chart[q_shear_free].clone()
            )
            block.attn.q_penultimate_weight = nn.Parameter(
                q_penultimate_chart[query_free].clone()
            )
            block.attn.q_target_weight = nn.Parameter(
                q_target_chart[query_free].clone()
            )
>>>>>>> REPLACE