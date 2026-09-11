MECHANISM: Independent scale gauge on the first head’s neighboring zero-bias query row

HYPOTHESIS: Adding the remaining diagonal query/key scale gauge to Reference Design 2’s qualified complementary first-head shear will reduce the model from 1487 to 1486 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified complementary terminal-row shear, normalize the adjacent zero-bias query row, omit its pivot coordinate, and transfer both its scale and the terminal-row shear into the corresponding key rows.

EVIDENCE: Reference Design 2 achieved 99.97% accuracy at 1487 parameters. Unlike the failed 1486 first-head mutual biased-row shear, this removes an independent scale degree of freedom from a zero-bias row, the same type of exact diagonal gauge already successful on the terminal row and both second-head zero-bias rows.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head's final zero-bias query row uses a
        # diagonal scale gauge. The second head's zero-bias rows use their
        # scale/shear gauge, one biased row is sheared against both, and the
        # other biased row is sheared against that freely biased anchor.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_anchor_weight = nn.Parameter(
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Both first-head zero-bias query rows use diagonal scale
        # gauges, with the terminal row additionally sheared against its
        # neighbor. The second head retains its qualified scale/shear gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_anchor_weight = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "q_first_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_anchor_pivot",
            torch.zeros((), dtype=torch.long),
        )
=======
        self.register_buffer(
            "q_first_neighbor_pivot",
            torch.zeros((), dtype=torch.long),
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
            "q_anchor_pivot",
            torch.zeros((), dtype=torch.long),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_first_pivot = int(self.q_first_pivot.item())
        q_anchor_pivot = int(self.q_anchor_pivot.item())
=======
        q_first_neighbor_pivot = int(
            self.q_first_neighbor_pivot.item()
        )
        q_first_pivot = int(self.q_first_pivot.item())
        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_anchor_pivot = int(self.q_anchor_pivot.item())
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_first_chart = torch.cat(
            (
                self.q_first_weight[:q_first_pivot],
                self.q_first_weight.new_full((1,), 1.0),
                self.q_first_weight[q_first_pivot:],
            )
        )
        q_first_relative = q_first_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_first_chart.norm()
        )
=======
        q_first_neighbor_chart = torch.cat(
            (
                self.q_first_neighbor_weight[
                    :q_first_neighbor_pivot
                ],
                self.q_first_neighbor_weight.new_full((1,), 1.0),
                self.q_first_neighbor_weight[
                    q_first_neighbor_pivot:
                ],
            )
        )
        q_first_neighbor_relative = q_first_neighbor_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_first_neighbor_chart.norm()
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_first_target = self.head_dim - 1
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_first_target],
                q_first_relative.unsqueeze(0),
                q_anchor_relative.unsqueeze(0),
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_first_target:],
            ),
            dim=0,
        )
=======
        q_first_neighbor = self.head_dim - 2
        q_first_target = self.head_dim - 1
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_first_neighbor],
                q_first_neighbor_relative.unsqueeze(0),
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
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_first_target = block.attn.head_dim - 1
            q_first_free = relative_qkv_weight[q_first_target]
            q_first_pivot = int(
                q_first_free.abs().argmax().item()
            )
            q_first_pivot_value = q_first_free[q_first_pivot]
            q_first_chart = q_first_free / q_first_pivot_value
            q_first_gauge_norm = (
                0.02 * math.sqrt(q_first_free.numel())
            )
            q_first_scale = (
                q_first_pivot_value.sign()
                * q_first_free.norm()
                / q_first_gauge_norm
            )
            key_first_target = cfg.d_model + q_first_target
            full_qkv_weight[key_first_target] = (
                q_first_scale * full_qkv_weight[key_first_target]
            )
            block.attn.q_first_pivot.fill_(q_first_pivot)
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

            q_first_neighbor_pivot = int(
                q_first_neighbor_free.abs().argmax().item()
            )
            q_first_neighbor_pivot_value = q_first_neighbor_free[
                q_first_neighbor_pivot
            ]
            q_first_neighbor_chart = (
                q_first_neighbor_free
                / q_first_neighbor_pivot_value
            )
            q_first_neighbor_gauge_norm = (
                0.02 * math.sqrt(q_first_neighbor_free.numel())
            )
            q_first_neighbor_scale = (
                q_first_neighbor_pivot_value.sign()
                * q_first_neighbor_free.norm()
                / q_first_neighbor_gauge_norm
            )
            q_first_neighbor_relative = (
                q_first_neighbor_chart
                * (
                    q_first_neighbor_gauge_norm
                    / q_first_neighbor_chart.norm()
                )
            )

            q_first_shear_pivot = q_first_neighbor_pivot
            q_first_shear = (
                q_first_free[q_first_shear_pivot]
                / q_first_neighbor_relative[q_first_shear_pivot]
            )
            q_first_sheared = (
                q_first_free
                - q_first_shear * q_first_neighbor_relative
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
            key_first_neighbor_free = full_qkv_weight[
                key_first_neighbor
            ].clone()
            key_first_target_free = full_qkv_weight[
                key_first_target
            ].clone()
            full_qkv_weight[key_first_neighbor] = (
                q_first_neighbor_scale * key_first_neighbor_free
                + q_first_shear * key_first_target_free
            )
            full_qkv_weight[key_first_target] = (
                q_first_scale * key_first_target_free
            )
            block.attn.q_first_neighbor_pivot.fill_(
                q_first_neighbor_pivot
            )
            block.attn.q_first_pivot.fill_(q_first_pivot)
            block.attn.q_first_shear_pivot.fill_(
                q_first_shear_pivot
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_first_target],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_first_weight = nn.Parameter(
                torch.cat(
                    (
                        q_first_chart[:q_first_pivot],
                        q_first_chart[q_first_pivot + 1:],
                    )
                )
            )
            block.attn.q_anchor_weight = nn.Parameter(
=======
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_first_neighbor],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_first_neighbor_weight = nn.Parameter(
                torch.cat(
                    (
                        q_first_neighbor_chart[
                            :q_first_neighbor_pivot
                        ],
                        q_first_neighbor_chart[
                            q_first_neighbor_pivot + 1:
                        ],
                    )
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
            block.attn.q_anchor_weight = nn.Parameter(
>>>>>>> REPLACE