MECHANISM: Orthogonal first-head zero-bias query/key gauge

HYPOTHESIS: Replacing the failed coordinate-aligned completion with a max-pivot-stable orthogonality chart will reduce the verified model from 1487 to 1486 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one coordinate from the first head’s remaining zero-bias query row, reconstruct it by enforcing orthogonality to the normalized terminal row, and apply the inverse initialization shear to the terminal key row.

EVIDENCE: The current complementary terminal-row shear achieved 99.97% at 1487 parameters, while the coordinate-aligned completion reached 97.76%; this tests the same remaining gauge with a balanced orthogonal constraint instead of another pivot-zero constraint.

<<<<<<< SEARCH
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_anchor_weight = nn.Parameter(
=======
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_anchor_weight = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_first_relative = q_first_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_first_chart.norm()
        )

        query_free = [
=======
        q_first_relative = q_first_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_first_chart.norm()
        )
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

        query_free = [
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
=======
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
            q_first_free_coordinates = [
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
            q_first_free_coordinates = [
>>>>>>> REPLACE