MECHANISM: Complementary first-head biased-row query/key shear

HYPOTHESIS: Applying the untested complementary mutual shear between the first head’s freely biased query rows atop the verified 1487-parameter design will produce 1486 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce Reference Design 1’s first-head terminal scale/shear chart, then omit one coordinate from the second biased query row using the leading biased row and compensate in the leading key row.

EVIDENCE: Reference Design 1 achieved 99.97% accuracy at 1487 parameters. The opposite first-head biased-row shear reached 91.85%, but the first-head zero-bias experiments showed strong directional asymmetry—the initially tested direction reached 97.09% while its complement reached 99.97%—making the untested complementary biased-row direction the closest distinct reduction.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The second head's zero-bias query rows use their full
        # scale/shear gauge. One bias-bearing row is sheared against both,
        # and the other is sheared against that freely biased row.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_anchor_weight = nn.Parameter(
            torch.empty(d_model - 2)
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
            "q_anchor_pivot",
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
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head's terminal zero-bias row uses its
        # qualified scale/shear chart, while its second biased row is sheared
        # against the leading biased row. The second head retains its full
        # qualified scale/shear construction.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_biased_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_anchor_weight = nn.Parameter(
            torch.empty(d_model - 2)
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
            "q_biased_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_anchor_pivot",
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
        q_anchor_pivot = int(self.q_anchor_pivot.item())
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
=======
        q_first_pivot = int(self.q_first_pivot.item())
        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_biased_pivot = int(self.q_biased_pivot.item())
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
        q_target_pivot = int(self.q_target_pivot.item())

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

        q_biased_relative = torch.cat(
            (
                self.q_biased_weight[:q_biased_pivot],
                self.q_biased_weight.new_zeros(1),
                self.q_biased_weight[q_biased_pivot:],
            )
        )

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

        q_anchor_relative = torch.cat(
            (
                self.q_anchor_weight[:q_anchor_pivot],
                self.q_anchor_weight.new_zeros(1),
                self.q_anchor_weight[q_anchor_pivot:],
            )
        )
        q_shear_relative = self.q_shear_weight.new_zeros(
            d_model - 1
        )
        q_shear_relative[query_free] = self.q_shear_weight

        q_biased_target = self.head_dim - 3
        q_first_target = self.head_dim - 1
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_biased_target],
                q_biased_relative.unsqueeze(0),
                self.qkv.weight[
                    q_biased_target:q_first_target - 1
                ],
                q_first_relative.unsqueeze(0),
                q_anchor_relative.unsqueeze(0),
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_first_target - 1:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_anchor = cfg.d_model - 4
            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
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

            key_first_neighbor = cfg.d_model + q_first_neighbor
            key_first_target = cfg.d_model + q_first_target
            key_first_target_free = full_qkv_weight[
                key_first_target
            ].clone()
            full_qkv_weight[key_first_neighbor] = (
                full_qkv_weight[key_first_neighbor]
                + q_first_shear * key_first_target_free
            )
            full_qkv_weight[key_first_target] = (
                q_first_scale * key_first_target_free
            )
            block.attn.q_first_pivot.fill_(q_first_pivot)
            block.attn.q_first_shear_pivot.fill_(
                q_first_shear_pivot
            )

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_biased_source = block.attn.head_dim - 4
            q_biased_target = block.attn.head_dim - 3
            q_biased_source_free = relative_qkv_weight[
                q_biased_source
            ]
            q_biased_target_free = relative_qkv_weight[
                q_biased_target
            ]
            q_biased_pivot = int(
                q_biased_source_free.abs().argmax().item()
            )
            q_biased_shear = (
                q_biased_target_free[q_biased_pivot]
                / q_biased_source_free[q_biased_pivot]
            )
            q_biased_chart = (
                q_biased_target_free
                - q_biased_shear * q_biased_source_free
            )
            q_biased_chart[q_biased_pivot] = 0.0

            key_biased_source = cfg.d_model + q_biased_source
            key_biased_target = cfg.d_model + q_biased_target
            key_biased_target_free = full_qkv_weight[
                key_biased_target
            ].clone()
            full_qkv_weight[key_biased_source] = (
                full_qkv_weight[key_biased_source]
                + q_biased_shear * key_biased_target_free
            )
            block.attn.q_biased_pivot.fill_(q_biased_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_anchor = cfg.d_model - 4
            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_biased_target],
                        relative_qkv_weight[
                            q_biased_target + 1:q_first_target
                        ],
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
            block.attn.q_biased_weight = nn.Parameter(
                torch.cat(
                    (
                        q_biased_chart[:q_biased_pivot],
                        q_biased_chart[q_biased_pivot + 1:],
                    )
                )
            )
            block.attn.q_anchor_weight = nn.Parameter(
>>>>>>> REPLACE