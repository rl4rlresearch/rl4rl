MECHANISM: Max-pivot normalization of the first head’s orthogonal zero-bias query row

HYPOTHESIS: Reproducing Reference Design 3’s normalized orthogonal query frame will reduce the current model from 1486 to 1485 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Normalize the first-head neighbor query with a fixed-norm max-pivot chart, absorb its initialization scale into the matching key row, and store one fewer learned coordinate.

EVIDENCE: Reference Design 3 used this exact reduction and achieved 99.92% accuracy with 1485 parameters, while every reported 1484-parameter extension failed the accuracy requirement or verification.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head's terminal zero-bias row uses a scale
        # gauge and the complementary shear against its zero-bias neighbor.
        # The second head uses its qualified scale and shear construction.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head's zero-bias query pair forms a
        # normalized orthogonal frame. The second head uses its qualified
        # scale and shear construction.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "q_first_shear_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_anchor_pivot",
            torch.zeros((), dtype=torch.long),
        )
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
            torch.zeros((), dtype=torch.long),
        )
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
            q_neighbor_shear = (
                q_first_neighbor_free * q_first_relative
            ).sum() / q_first_relative.square().sum()
            q_neighbor_chart = (
                q_first_neighbor_free
                - q_neighbor_shear * q_first_relative
            )

            key_first_neighbor = cfg.d_model + q_first_neighbor
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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