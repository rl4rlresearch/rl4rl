MECHANISM: Max-pivot projected orthogonality gauge for the first-head biased query pair

HYPOTHESIS: Extending the verified 1485-parameter normalized orthogonal-frame design with a balanced orthogonality chart for the first head’s freely biased query pair will reduce the model to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce Reference Design 3, then omit one coordinate from the second biased first-head query row, reconstruct it by projection orthogonal to the first biased row, and apply the inverse initialization shear to the matching key row.

EVIDENCE: Reference Design 3 reached 99.92% at 1485 parameters. Direct coordinate shears between the first-head biased rows failed, whereas replacing a coordinate-aligned first-head zero-bias completion with a max-pivot orthogonality chart raised accuracy from 97.76% to 99.79%, and normalizing that orthogonal frame reached 99.92%.

<<<<<<< SEARCH
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
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head uses normalized orthogonal zero-bias
        # rows and a projected orthogonality chart for its freely biased pair.
        # The second head uses its qualified scale and shear construction.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_biased_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 3)
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
            "q_biased_pivot",
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
            "q_neighbor_pivot",
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
=======
        q_biased_pivot = int(self.q_biased_pivot.item())
        q_first_pivot = int(self.q_first_pivot.item())
        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_neighbor_pivot = int(self.q_neighbor_pivot.item())
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
        q_target_pivot = int(self.q_target_pivot.item())
        query_free = [
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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

        q_biased_first = self.head_dim - 4
        q_biased_second = self.head_dim - 3
        q_biased_reference = self.qkv.weight[q_biased_first]
        q_biased_coordinates = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate != q_biased_pivot
        ]
        q_biased_lift = self.q_biased_weight.new_zeros(
            d_model - 1
        )
        q_biased_lift[q_biased_coordinates] = (
            self.q_biased_weight
        )
        q_biased_relative = (
            q_biased_lift
            - q_biased_reference
            * (
                (q_biased_lift * q_biased_reference).sum()
                / q_biased_reference.square().sum()
            )
        )

        q_shear_relative = self.q_shear_weight.new_zeros(
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        q_first_neighbor = self.head_dim - 2
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_biased_second],
                q_biased_relative.unsqueeze(0),
                q_neighbor_relative.unsqueeze(0),
                q_first_relative.unsqueeze(0),
                q_anchor_relative.unsqueeze(0),
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_biased_second:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_anchor = cfg.d_model - 4
=======
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_biased_first = block.attn.head_dim - 4
            q_biased_second = block.attn.head_dim - 3
            q_biased_reference = relative_qkv_weight[
                q_biased_first
            ]
            q_biased_free = relative_qkv_weight[
                q_biased_second
            ]
            q_biased_shear = (
                q_biased_free * q_biased_reference
            ).sum() / q_biased_reference.square().sum()
            q_biased_orthogonal = (
                q_biased_free
                - q_biased_shear * q_biased_reference
            )
            q_biased_pivot = int(
                q_biased_reference.abs().argmax().item()
            )
            q_biased_lift = (
                q_biased_orthogonal
                - (
                    q_biased_orthogonal[q_biased_pivot]
                    / q_biased_reference[q_biased_pivot]
                )
                * q_biased_reference
            )
            q_biased_lift[q_biased_pivot] = 0.0

            key_biased_first = cfg.d_model + q_biased_first
            key_biased_second = cfg.d_model + q_biased_second
            key_biased_second_free = full_qkv_weight[
                key_biased_second
            ].clone()
            full_qkv_weight[key_biased_first] = (
                full_qkv_weight[key_biased_first]
                + q_biased_shear * key_biased_second_free
            )
            block.attn.q_biased_pivot.fill_(q_biased_pivot)

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
                        relative_qkv_weight[:q_biased_second],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            q_biased_coordinates = [
                coordinate
                for coordinate in range(q_biased_lift.numel())
                if coordinate != q_biased_pivot
            ]
            block.attn.q_biased_weight = nn.Parameter(
                q_biased_lift[q_biased_coordinates].clone()
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
            block.attn.q_anchor_weight = nn.Parameter(
>>>>>>> REPLACE