MECHANISM: Max-pivot first-head biased-query shear atop the qualified orthogonal frame

HYPOTHESIS: Completing Reference Design 3’s verified normalization and applying the second head’s successful biased-query coordinate shear to the first head will reduce the model from 1486 to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Normalize the first head’s orthogonal zero-bias neighbor, then omit one coordinate from its first biased query row and transfer the initialization shear into the paired key row.

EVIDENCE: Reference Design 3 achieved 99.92% at 1485 parameters, and Reference Design 1 achieved 99.99% with the analogous max-pivot shear between two biased queries. Unlike failed orthogonality and affine-normalization attempts, this adds only an exact coordinate Q/K basis shear.

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
        # output bias. The first head uses a normalized orthogonal zero-bias
        # frame and a max-pivot shear between its two biased query rows. The
        # second head uses its qualified scale and shear construction.
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
            "q_biased_pivot",
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
        q_biased_pivot = int(self.q_biased_pivot.item())
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

        q_biased_relative = torch.cat(
            (
                self.q_biased_weight[:q_biased_pivot],
                self.q_biased_weight.new_zeros(1),
                self.q_biased_weight[q_biased_pivot:],
            )
        )

        query_free = [
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        q_first_neighbor = self.head_dim - 2
        qkv_rows = torch.cat(
            (
                q_biased_relative.unsqueeze(0),
                self.qkv.weight[:q_first_neighbor - 1],
                q_neighbor_relative.unsqueeze(0),
                q_first_relative.unsqueeze(0),
                q_anchor_relative.unsqueeze(0),
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_first_neighbor - 1:],
            ),
            dim=0,
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

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_anchor = cfg.d_model - 4
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

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_biased = 0
            q_biased_reference = 1
            q_biased_free = relative_qkv_weight[q_biased]
            q_biased_reference_free = relative_qkv_weight[
                q_biased_reference
            ]
            q_biased_pivot = int(
                q_biased_reference_free.abs().argmax().item()
            )
            q_biased_shear = (
                q_biased_free[q_biased_pivot]
                / q_biased_reference_free[q_biased_pivot]
            )
            q_biased_chart = (
                q_biased_free
                - q_biased_shear * q_biased_reference_free
            )
            q_biased_chart[q_biased_pivot] = 0.0

            key_biased = cfg.d_model + q_biased
            key_biased_reference = (
                cfg.d_model + q_biased_reference
            )
            full_qkv_weight[key_biased_reference] = (
                full_qkv_weight[key_biased_reference]
                + q_biased_shear
                * full_qkv_weight[key_biased]
            )
            block.attn.q_biased_pivot.fill_(q_biased_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_anchor = cfg.d_model - 4
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
=======
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[
                            q_biased + 1:q_first_neighbor
                        ],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_biased_weight = nn.Parameter(
                torch.cat(
                    (
                        q_biased_chart[:q_biased_pivot],
                        q_biased_chart[q_biased_pivot + 1:],
                    )
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
>>>>>>> REPLACE