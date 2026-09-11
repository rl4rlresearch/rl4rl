MECHANISM: Max-pivot complementary biased-query shear

HYPOTHESIS: Completing the second head’s biased-query basis reduction with a conditioned complementary shear will reduce the verified 1485-parameter model to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one coordinate from the second biased query after reducing the anchor modulo the two zero-bias queries, and apply the exact inverse transformation to the corresponding key rows.

EVIDENCE: The current max-pivot query-frame design achieved 99.92% at 1485 parameters, and Reference Design 1 achieved 99.99% with the first shear between these biased queries. Unlike failed normalization, bias tying, and geometric constraints, this retains both independent biases and removes another continuous query/key basis redundancy.

<<<<<<< SEARCH
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
=======
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "q_anchor_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_penultimate_pivot",
            torch.zeros((), dtype=torch.long),
        )
=======
        self.register_buffer(
            "q_anchor_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_shear_pair_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_penultimate_pivot",
            torch.zeros((), dtype=torch.long),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
=======
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_shear_pair_pivot = int(
            self.q_shear_pair_pivot.item()
        )
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_shear_relative = self.q_shear_weight.new_zeros(
            d_model - 1
        )
        q_shear_relative[query_free] = self.q_shear_weight
=======
        q_shear_free = [
            coordinate
            for coordinate in query_free
            if coordinate != q_shear_pair_pivot
        ]
        q_shear_relative = self.q_shear_weight.new_zeros(
            d_model - 1
        )
        q_shear_relative[q_shear_free] = self.q_shear_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
            q_anchor_chart = (
                q_anchor_free - anchor_shear * q_shear_chart
            )
            q_anchor_chart[q_anchor_pivot] = 0.0

            key_anchor = cfg.d_model + q_anchor
=======
            q_anchor_chart = (
                q_anchor_free - anchor_shear * q_shear_chart
            )
            q_anchor_chart[q_anchor_pivot] = 0.0

            q_anchor_penultimate_shear = (
                q_anchor_chart[q_penultimate_pivot]
                / q_penultimate_relative[q_penultimate_pivot]
            )
            q_anchor_reduced = (
                q_anchor_chart
                - q_anchor_penultimate_shear
                * q_penultimate_relative
            )
            q_anchor_target_shear = (
                q_anchor_reduced[q_target_pivot]
                / q_target_relative[q_target_pivot]
            )
            q_anchor_reduced = (
                q_anchor_reduced
                - q_anchor_target_shear * q_target_relative
            )
            q_anchor_reduced[q_penultimate_pivot] = 0.0
            q_anchor_reduced[q_target_pivot] = 0.0
            q_shear_pair_candidates = [
                coordinate
                for coordinate in range(q_anchor_reduced.numel())
                if coordinate
                not in (q_penultimate_pivot, q_target_pivot)
            ]
            q_shear_pair_pivot = max(
                q_shear_pair_candidates,
                key=lambda coordinate: abs(
                    q_anchor_reduced[coordinate].item()
                ),
            )
            q_shear_pair = (
                q_shear_chart[q_shear_pair_pivot]
                / q_anchor_reduced[q_shear_pair_pivot]
            )
            q_shear_chart = (
                q_shear_chart
                - q_shear_pair * q_anchor_reduced
            )
            q_shear_chart[q_penultimate_pivot] = 0.0
            q_shear_chart[q_target_pivot] = 0.0
            q_shear_chart[q_shear_pair_pivot] = 0.0

            key_anchor = cfg.d_model + q_anchor
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            key_shear_sheared = (
                key_shear_free + anchor_shear * key_anchor_free
            )
            full_qkv_weight[key_anchor] = (
                key_anchor_free
                + q_shear_pair * key_shear_sheared
            )
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale * key_penultimate_sheared
                + bias_penultimate_shear * key_shear_free
                - q_shear_pair
                * q_anchor_penultimate_shear
                * key_shear_sheared
            )
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + bias_target_shear * key_shear_free
                - q_shear_pair
                * q_anchor_target_shear
                * key_shear_sheared
            )
            full_qkv_weight[key_shear] = key_shear_sheared
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.q_anchor_pivot.fill_(q_anchor_pivot)
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
            )
=======
            block.attn.q_anchor_pivot.fill_(q_anchor_pivot)
            block.attn.q_shear_pair_pivot.fill_(
                q_shear_pair_pivot
            )
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.q_shear_weight = nn.Parameter(
                q_shear_chart[query_free].clone()
            )
            block.attn.q_penultimate_weight = nn.Parameter(
=======
            q_shear_free_coordinates = [
                coordinate
                for coordinate in query_free
                if coordinate != q_shear_pair_pivot
            ]
            block.attn.q_shear_weight = nn.Parameter(
                q_shear_chart[q_shear_free_coordinates].clone()
            )
            block.attn.q_penultimate_weight = nn.Parameter(
>>>>>>> REPLACE