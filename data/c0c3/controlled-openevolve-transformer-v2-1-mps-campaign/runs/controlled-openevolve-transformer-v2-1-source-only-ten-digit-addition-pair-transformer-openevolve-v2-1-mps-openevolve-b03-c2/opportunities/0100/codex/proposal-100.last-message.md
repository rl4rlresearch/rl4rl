MECHANISM: Max-pivot normalization of the sheared second-head anchor query

HYPOTHESIS: Normalizing the already-sheared second-head anchor query and transferring its scale into the matching key row will reduce Reference Design 3 from 1485 to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified 1485-parameter normalized orthogonal-query design, then represent the second-head anchor row with a fixed-norm max-pivot chart and absorb its initialization scale into the corresponding key row.

EVIDENCE: Reference Design 3 achieved 99.92% after max-pivot normalization of a sheared query row. The failed 1484 key-row normalization used the opposite query/key orientation, so query-side normalization of the already-qualified anchor shear is the closest distinct untested gauge.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final two query-bias
        # coordinates in each head. The second head's trailing query rows use
        # their full scale/shear gauge, and its neighboring bias-bearing row
        # uses an additional bias-preserving shear against the target row.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.register_buffer(
            "q_penultimate_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_target_pivot",
            torch.zeros((), dtype=torch.long),
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head's zero-bias query pair forms a
        # normalized orthogonal frame. The second head's zero-bias rows use
        # their scale/shear gauge, and its two biased rows use conditioned
        # shears. The final anchor row is normalized after its shear.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_anchor_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.register_buffer(
            "q_first_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_first_shear_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_neighbor_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_anchor_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_anchor_scale_pivot",
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

        q_shear_relative = torch.cat(
            (
                self.q_shear_weight[:q_target_pivot],
                self.q_shear_weight.new_zeros(1),
                self.q_shear_weight[q_target_pivot:],
            )
        )
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
=======
        q_first_pivot = int(self.q_first_pivot.item())
        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_neighbor_pivot = int(self.q_neighbor_pivot.item())
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_anchor_scale_pivot = int(
            self.q_anchor_scale_pivot.item()
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

        q_first_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate
            not in (q_first_pivot, q_first_shear_pivot)
        ]
        q_first_chart = self.q_first_weight.new_zeros(
            d_model - 1
        )
        q_first_chart[q_first_pivot] = 1.0
        q_first_chart[q_first_free] = self.q_first_weight
        q_first_relative = q_first_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_first_chart.norm()
        )

        q_neighbor_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate
            not in (q_first_pivot, q_neighbor_pivot)
        ]
        q_neighbor_chart = self.q_neighbor_weight.new_zeros(
            d_model - 1
        )
        q_neighbor_chart[q_neighbor_pivot] = 1.0
        q_neighbor_chart[q_neighbor_free] = (
            self.q_neighbor_weight
        )
        q_neighbor_coordinates = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate != q_first_pivot
        ]
        q_neighbor_chart[q_first_pivot] = -(
            q_neighbor_chart[q_neighbor_coordinates]
            * q_first_relative[q_neighbor_coordinates]
        ).sum() / q_first_relative[q_first_pivot]
        q_neighbor_relative = q_neighbor_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_neighbor_chart.norm()
        )

        q_anchor_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate
            not in (q_anchor_pivot, q_anchor_scale_pivot)
        ]
        q_anchor_chart = self.q_anchor_weight.new_zeros(
            d_model - 1
        )
        q_anchor_chart[q_anchor_scale_pivot] = 1.0
        q_anchor_chart[q_anchor_free] = self.q_anchor_weight
        q_anchor_relative = q_anchor_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_anchor_chart.norm()
        )

        q_shear_relative = self.q_shear_weight.new_zeros(
            d_model - 1
        )
        q_shear_relative[query_free] = self.q_shear_weight

        q_first_neighbor = self.head_dim - 2
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_first_neighbor],
                q_neighbor_relative.unsqueeze(0),
                q_first_relative.unsqueeze(0),
                q_anchor_relative.unsqueeze(0),
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_first_neighbor:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. Two conditioned value/output shears are
        # fixed per head. A full two-coordinate query/key gauge then fixes
        # both scales and both off-diagonal shears in the second head.
=======
        # Store LayerNorm-null input and residual-output directions through
        # relative representatives. The first head uses a normalized
        # orthogonal query frame. The second head uses its qualified query
        # shears, with an additional max-pivot anchor normalization.
>>>>>>> REPLACE

<<<<<<< SEARCH
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
            q_shear_free = relative_qkv_weight[q_shear]
            q_penultimate_free = relative_qkv_weight[q_penultimate]
            q_target_free = relative_qkv_weight[q_target]
=======
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_first_neighbor = block.attn.head_dim - 2
            q_first_target = block.attn.head_dim - 1
            q_first_neighbor_free = relative_qkv_weight[
                q_first_neighbor
            ]
            q_first_free = relative_qkv_weight[q_first_target]

            q_first_shear_pivot = int(
                q_first_neighbor_free.abs().argmax().item()
            )
            q_first_shear = (
                q_first_free[q_first_shear_pivot]
                / q_first_neighbor_free[q_first_shear_pivot]
            )
            q_first_sheared = (
                q_first_free
                - q_first_shear * q_first_neighbor_free
            )
            q_first_sheared[q_first_shear_pivot] = 0.0

            q_first_pivot = int(
                q_first_sheared.abs().argmax().item()
            )
            q_first_pivot_value = q_first_sheared[q_first_pivot]
            q_first_chart = (
                q_first_sheared / q_first_pivot_value
            )
            q_first_gauge_norm = (
                0.02 * math.sqrt(q_first_sheared.numel())
            )
            q_first_scale = (
                q_first_pivot_value.sign()
                * q_first_sheared.norm()
                / q_first_gauge_norm
            )
            q_first_relative = q_first_chart * (
                q_first_gauge_norm / q_first_chart.norm()
            )

            q_neighbor_shear = (
                q_first_neighbor_free * q_first_relative
            ).sum() / q_first_relative.square().sum()
            q_neighbor_orthogonal = (
                q_first_neighbor_free
                - q_neighbor_shear * q_first_relative
            )
            q_neighbor_candidates = [
                coordinate
                for coordinate in range(
                    q_neighbor_orthogonal.numel()
                )
                if coordinate != q_first_pivot
            ]
            q_neighbor_pivot = max(
                q_neighbor_candidates,
                key=lambda coordinate: abs(
                    q_neighbor_orthogonal[coordinate].item()
                ),
            )
            q_neighbor_pivot_value = q_neighbor_orthogonal[
                q_neighbor_pivot
            ]
            q_neighbor_chart = (
                q_neighbor_orthogonal / q_neighbor_pivot_value
            )
            q_neighbor_gauge_norm = (
                0.02 * math.sqrt(q_neighbor_orthogonal.numel())
            )
            q_neighbor_scale = (
                q_neighbor_pivot_value.sign()
                * q_neighbor_orthogonal.norm()
                / q_neighbor_gauge_norm
            )

            key_first_neighbor = cfg.d_model + q_first_neighbor
            key_first_target = cfg.d_model + q_first_target
            key_first_neighbor_free = full_qkv_weight[
                key_first_neighbor
            ].clone()
            key_first_target_free = full_qkv_weight[
                key_first_target
            ].clone()
            key_first_neighbor_sheared = (
                key_first_neighbor_free
                + q_first_shear * key_first_target_free
            )
            full_qkv_weight[key_first_neighbor] = (
                q_neighbor_scale * key_first_neighbor_sheared
            )
            full_qkv_weight[key_first_target] = (
                q_first_scale * key_first_target_free
                + q_neighbor_shear * key_first_neighbor_sheared
            )
            block.attn.q_first_pivot.fill_(q_first_pivot)
            block.attn.q_first_shear_pivot.fill_(
                q_first_shear_pivot
            )
            block.attn.q_neighbor_pivot.fill_(q_neighbor_pivot)

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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            q_shear_coefficient = (
                q_shear_free[q_target_pivot]
                / q_target_relative[q_target_pivot]
            )
            q_shear_chart = (
                q_shear_free
                - q_shear_coefficient * q_target_relative
            )
            q_shear_chart[q_target_pivot] = 0.0
=======
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
            q_anchor_sheared = (
                q_anchor_free - anchor_shear * q_shear_chart
            )
            q_anchor_sheared[q_anchor_pivot] = 0.0
            q_anchor_scale_candidates = [
                coordinate
                for coordinate in range(q_anchor_sheared.numel())
                if coordinate != q_anchor_pivot
            ]
            q_anchor_scale_pivot = max(
                q_anchor_scale_candidates,
                key=lambda coordinate: abs(
                    q_anchor_sheared[coordinate].item()
                ),
            )
            q_anchor_scale_pivot_value = q_anchor_sheared[
                q_anchor_scale_pivot
            ]
            q_anchor_chart = (
                q_anchor_sheared / q_anchor_scale_pivot_value
            )
            q_anchor_gauge_norm = (
                0.02 * math.sqrt(q_anchor_sheared.numel())
            )
            q_anchor_scale = (
                q_anchor_scale_pivot_value.sign()
                * q_anchor_sheared.norm()
                / q_anchor_gauge_norm
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                + q_shear_coefficient
                * full_qkv_weight[key_shear]
            )
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
            )
            block.attn.q_target_pivot.fill_(q_target_pivot)
=======
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
            full_qkv_weight[key_anchor] = (
                q_anchor_scale * key_anchor_free
            )
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale * key_penultimate_sheared
                + bias_penultimate_shear * key_shear_free
            )
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + bias_target_shear * key_shear_free
            )
            full_qkv_weight[key_shear] = (
                key_shear_free + anchor_shear * key_anchor_free
            )
            block.attn.q_anchor_pivot.fill_(q_anchor_pivot)
            block.attn.q_anchor_scale_pivot.fill_(
                q_anchor_scale_pivot
            )
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
            )
            block.attn.q_target_pivot.fill_(q_target_pivot)
>>>>>>> REPLACE

<<<<<<< SEARCH
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_shear],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
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
=======
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_first_neighbor],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            q_first_free_coordinates = [
                coordinate
                for coordinate in range(q_first_chart.numel())
                if coordinate
                not in (q_first_pivot, q_first_shear_pivot)
            ]
            block.attn.q_first_weight = nn.Parameter(
                q_first_chart[q_first_free_coordinates].clone()
            )
            q_neighbor_free_coordinates = [
                coordinate
                for coordinate in range(q_neighbor_chart.numel())
                if coordinate
                not in (q_first_pivot, q_neighbor_pivot)
            ]
            block.attn.q_neighbor_weight = nn.Parameter(
                q_neighbor_chart[
                    q_neighbor_free_coordinates
                ].clone()
            )
            q_anchor_free_coordinates = [
                coordinate
                for coordinate in range(q_anchor_chart.numel())
                if coordinate
                not in (q_anchor_pivot, q_anchor_scale_pivot)
            ]
            block.attn.q_anchor_weight = nn.Parameter(
                q_anchor_chart[q_anchor_free_coordinates].clone()
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