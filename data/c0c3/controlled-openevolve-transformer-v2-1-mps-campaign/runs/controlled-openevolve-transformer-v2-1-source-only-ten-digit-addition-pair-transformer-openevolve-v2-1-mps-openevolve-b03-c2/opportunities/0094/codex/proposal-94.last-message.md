MECHANISM: Max-pivot diagonal value/output gauge on the second-head neighboring row

HYPOTHESIS: Adding one direction-preserving second-head value/output scale gauge to the verified 1485-parameter normalized orthogonal-query design will yield 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Complete the qualified first-head query-frame normalization, then normalize the second head’s penultimate projection row and transfer its scale into the matching value row; update projection reconstruction used by training.

EVIDENCE: Reference Design 3 achieved 99.92% at 1485 parameters, and identical diagonal value/output gauges already succeed on both target rows. The 1484-parameter neighboring-row orthogonality attempt reached 97.24%; preserving the row’s direction while removing only its scale is a less restrictive test of the remaining gauge.

<<<<<<< SEARCH
        self.q_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_anchor_weight = nn.Parameter(
=======
        self.q_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_anchor_weight = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "q_first_shear_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_anchor_pivot",
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.register_buffer(
            "proj_head_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "proj_last_pivot",
=======
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.proj_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.register_buffer(
            "proj_head_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "proj_neighbor_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "proj_last_pivot",
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_anchor_pivot = int(self.q_anchor_pivot.item())
=======
        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_neighbor_pivot = int(self.q_neighbor_pivot.item())
        q_anchor_pivot = int(self.q_anchor_pivot.item())
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_neighbor_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate != q_first_pivot
        ]
        q_neighbor_relative = self.q_neighbor_weight.new_zeros(
            d_model - 1
        )
        q_neighbor_relative[q_neighbor_free] = (
            self.q_neighbor_weight
        )
        q_neighbor_relative[q_first_pivot] = -(
            q_neighbor_relative[q_neighbor_free]
            * q_first_relative[q_neighbor_free]
        ).sum() / q_first_relative[q_first_pivot]
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_free = last_chart * (
            (0.02 * math.sqrt(d_model - 3))
            / last_chart.norm()
        )
        head_relative = torch.cat(
=======
        last_free = last_chart * (
            (0.02 * math.sqrt(d_model - 3))
            / last_chart.norm()
        )
        neighbor_pivot = int(self.proj_neighbor_pivot.item())
        neighbor_chart = torch.cat(
            (
                self.proj_neighbor_weight[:neighbor_pivot],
                self.proj_neighbor_weight.new_full((1,), 1.0),
                self.proj_neighbor_weight[neighbor_pivot:],
            )
        )
        neighbor_free = neighbor_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / neighbor_chart.norm()
        )
        head_relative = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.proj.weight[:split],
                head_relative.unsqueeze(0),
                self.proj.weight[split:],
                last_relative.unsqueeze(0),
=======
                self.proj.weight[:split],
                head_relative.unsqueeze(0),
                self.proj.weight[split:],
                neighbor_free.unsqueeze(0),
                last_relative.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.proj_last_pivot.fill_(last_pivot)

            relative_qkv_weight = (
=======
            block.attn.proj_last_pivot.fill_(last_pivot)

            neighbor_row = cfg.d_model - 2
            neighbor_free = relative_proj_weight[neighbor_row]
            neighbor_pivot = int(
                neighbor_free.abs().argmax().item()
            )
            neighbor_pivot_value = neighbor_free[neighbor_pivot]
            neighbor_chart = (
                neighbor_free / neighbor_pivot_value
            )
            neighbor_gauge_norm = (
                0.02 * math.sqrt(neighbor_free.numel())
            )
            neighbor_scale = (
                neighbor_pivot_value.sign()
                * neighbor_free.norm()
                / neighbor_gauge_norm
            )
            full_qkv_weight[value_start + neighbor_row] = (
                neighbor_scale
                * full_qkv_weight[value_start + neighbor_row]
            )
            block.attn.proj_neighbor_pivot.fill_(neighbor_pivot)

            relative_qkv_weight = (
>>>>>>> REPLACE

<<<<<<< SEARCH
            q_neighbor_shear = (
                q_first_neighbor_free * q_first_relative
            ).sum() / q_first_relative.square().sum()
            q_neighbor_chart = (
                q_first_neighbor_free
                - q_neighbor_shear * q_first_relative
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
                + q_neighbor_shear
                * full_qkv_weight[key_first_neighbor]
            )
            block.attn.q_first_pivot.fill_(q_first_pivot)
            block.attn.q_first_shear_pivot.fill_(
                q_first_shear_pivot
            )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            q_neighbor_free_coordinates = [
                coordinate
                for coordinate in range(q_neighbor_chart.numel())
                if coordinate != q_first_pivot
            ]
            block.attn.q_neighbor_weight = nn.Parameter(
                q_neighbor_chart[
                    q_neighbor_free_coordinates
                ].clone()
            )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            block.attn.proj.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_proj_weight[:first_target],
                        relative_proj_weight[
                            first_target + 1:neighbor_row
                        ],
                    ),
                    dim=0,
                )
            )
            block.attn.proj_neighbor_weight = nn.Parameter(
                torch.cat(
                    (
                        neighbor_chart[:neighbor_pivot],
                        neighbor_chart[neighbor_pivot + 1:],
                    )
                )
            )
            block.attn.proj_head_weight = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
def reconstruct_attention_output_weight(
    weight,
    head_weight,
    last_weight,
    head_dim,
    head_pivot,
    last_pivot,
):
=======
def reconstruct_attention_output_weight(
    weight,
    head_weight,
    neighbor_weight,
    last_weight,
    head_dim,
    head_pivot,
    neighbor_pivot,
    last_pivot,
):
>>>>>>> REPLACE

<<<<<<< SEARCH
    last_free = last_chart * (
        (0.02 * math.sqrt(last_weight.numel() + 1))
        / last_chart.norm()
    )
    head_relative = torch.cat(
=======
    last_free = last_chart * (
        (0.02 * math.sqrt(last_weight.numel() + 1))
        / last_chart.norm()
    )
    neighbor_chart = torch.cat(
        (
            neighbor_weight[:neighbor_pivot],
            neighbor_weight.new_full((1,), 1.0),
            neighbor_weight[neighbor_pivot:],
        )
    )
    neighbor_free = neighbor_chart * (
        (0.02 * math.sqrt(neighbor_weight.numel() + 1))
        / neighbor_chart.norm()
    )
    head_relative = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
            weight[:split],
            head_relative.unsqueeze(0),
            weight[split:],
            last_relative.unsqueeze(0),
=======
            weight[:split],
            head_relative.unsqueeze(0),
            weight[split:],
            neighbor_free.unsqueeze(0),
            last_relative.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_neighbor_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            head_pivot,
            neighbor_pivot,
            last_pivot,
        ) in self.value_bias_specs:
>>>>>>> REPLACE

<<<<<<< SEARCH
                    reconstruct_attention_output_weight(
                        proj_weight.detach(),
                        proj_head_weight.detach(),
                        proj_last_weight.detach(),
                        head_dim,
                        head_pivot,
                        last_pivot,
                    )
=======
                    reconstruct_attention_output_weight(
                        proj_weight.detach(),
                        proj_head_weight.detach(),
                        proj_neighbor_weight.detach(),
                        proj_last_weight.detach(),
                        head_dim,
                        head_pivot,
                        neighbor_pivot,
                        last_pivot,
                    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Update the omitted value-bias coordinate in full-coordinate AdamW,
        # then immediately return to the zero-coordinate gauge by folding its
        # effect into the already-updated attention projection bias.
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
=======
        # Update the omitted value-bias coordinate in full-coordinate AdamW,
        # then immediately return to the zero-coordinate gauge by folding its
        # effect into the already-updated attention projection bias.
        for (
            qkv_bias,
            proj_weight,
            proj_head_weight,
            proj_neighbor_weight,
            proj_last_weight,
            proj_bias,
            head_dim,
            head_pivot,
            neighbor_pivot,
            last_pivot,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
>>>>>>> REPLACE

<<<<<<< SEARCH
                reconstruct_attention_output_weight(
                    proj_weight,
                    proj_head_weight,
                    proj_last_weight,
                    head_dim,
                    head_pivot,
                    last_pivot,
                )
=======
                reconstruct_attention_output_weight(
                    proj_weight,
                    proj_head_weight,
                    proj_neighbor_weight,
                    proj_last_weight,
                    head_dim,
                    head_pivot,
                    neighbor_pivot,
                    last_pivot,
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
        head_pivot,
        last_pivot,
    ) in value_bias_specs:
=======
    for (
        qkv_bias,
        proj_weight,
        proj_head_weight,
        proj_neighbor_weight,
        proj_last_weight,
        proj_bias,
        head_dim,
        head_pivot,
        neighbor_pivot,
        last_pivot,
    ) in value_bias_specs:
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_proj_weight = reconstruct_attention_output_weight(
            proj_weight.detach(),
            proj_head_weight.detach(),
            proj_last_weight.detach(),
            head_dim,
            head_pivot,
            last_pivot,
        )
=======
        full_proj_weight = reconstruct_attention_output_weight(
            proj_weight.detach(),
            proj_head_weight.detach(),
            proj_neighbor_weight.detach(),
            proj_last_weight.detach(),
            head_dim,
            head_pivot,
            neighbor_pivot,
            last_pivot,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_head_weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
            int(block.attn.proj_head_pivot.item()),
            int(block.attn.proj_last_pivot.item()),
        )
=======
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj_head_weight,
            block.attn.proj_neighbor_weight,
            block.attn.proj_last_weight,
            block.attn.proj.bias,
            block.attn.head_dim,
            int(block.attn.proj_head_pivot.item()),
            int(block.attn.proj_neighbor_pivot.item()),
            int(block.attn.proj_last_pivot.item()),
        )
>>>>>>> REPLACE