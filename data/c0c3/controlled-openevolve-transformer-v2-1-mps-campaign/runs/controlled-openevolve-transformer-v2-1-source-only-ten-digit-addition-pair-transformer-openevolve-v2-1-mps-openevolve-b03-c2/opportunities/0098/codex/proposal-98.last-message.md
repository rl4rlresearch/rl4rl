MECHANISM: Max-pivot diagonal gauge on the first head’s biased-coordinate key row

HYPOTHESIS: Normalizing the first head’s unconstrained biased-coordinate key row atop the verified normalized orthogonal-query frame will reduce the model from 1485 to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Complete Reference Design 3’s normalization of the first-head zero-bias query frame, then remove the scale of the first biased-coordinate key row and transfer its initialization scale into the matching query row and bias.

EVIDENCE: Reference Design 3 achieved 99.92% at 1485 parameters. Prior 1484 failures constrained query shears or value/output rows; this tests an untried independent diagonal query/key gauge while leaving the biased query direction unconstrained.

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
        self.q_anchor_weight = nn.Parameter(
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head's zero-bias query pair forms a
        # normalized orthogonal frame, and its first biased-coordinate key
        # row uses an independent diagonal scale gauge. The second head uses
        # its qualified scale and shear construction.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_neighbor_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.k_first_weight = nn.Parameter(
            torch.empty(d_model - 2)
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
            "k_first_pivot",
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
        k_first_pivot = int(self.k_first_pivot.item())
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

        query_free = [
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

        k_first_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate != k_first_pivot
        ]
        k_first_chart = self.k_first_weight.new_zeros(
            d_model - 1
        )
        k_first_chart[k_first_pivot] = 1.0
        k_first_chart[k_first_free] = self.k_first_weight
        k_first_relative = k_first_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / k_first_chart.norm()
        )

        query_free = [
>>>>>>> REPLACE

<<<<<<< SEARCH
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_first_neighbor:],
=======
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                k_first_relative.unsqueeze(0),
                self.qkv.weight[q_first_neighbor:],
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

            k_first = cfg.d_model
            k_first_free = (
                full_qkv_weight[k_first, :-1]
                - full_qkv_weight[k_first, -1:]
            )
            k_first_pivot = int(
                k_first_free.abs().argmax().item()
            )
            k_first_pivot_value = k_first_free[k_first_pivot]
            k_first_chart = k_first_free / k_first_pivot_value
            k_first_gauge_norm = (
                0.02 * math.sqrt(k_first_free.numel())
            )
            k_first_scale = (
                k_first_pivot_value.sign()
                * k_first_free.norm()
                / k_first_gauge_norm
            )
            full_qkv_weight[0] = (
                k_first_scale * full_qkv_weight[0]
            )
            with torch.no_grad():
                block.attn.qkv.bias[0].mul_(k_first_scale)
            block.attn.k_first_pivot.fill_(k_first_pivot)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                coordinate
                for coordinate in range(q_first_chart.numel())
                if coordinate
                not in (q_first_pivot, q_first_shear_pivot)
            ]
            block.attn.q_first_weight = nn.Parameter(
                q_first_chart[q_first_free_coordinates].clone()
            )
            block.attn.q_anchor_weight = nn.Parameter(
=======
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_first_neighbor],
                        relative_qkv_weight[k_first + 1:],
                    ),
                    dim=0,
                )
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
            q_first_free_coordinates = [
                coordinate
                for coordinate in range(q_first_chart.numel())
                if coordinate
                not in (q_first_pivot, q_first_shear_pivot)
            ]
            block.attn.q_first_weight = nn.Parameter(
                q_first_chart[q_first_free_coordinates].clone()
            )
            k_first_free_coordinates = [
                coordinate
                for coordinate in range(k_first_chart.numel())
                if coordinate != k_first_pivot
            ]
            block.attn.k_first_weight = nn.Parameter(
                k_first_chart[k_first_free_coordinates].clone()
            )
            block.attn.q_anchor_weight = nn.Parameter(
>>>>>>> REPLACE