MECHANISM: Penultimate-first second-head anchor shear

HYPOTHESIS: Shearing the remaining second-head biased anchor row against the normalized penultimate zero-bias row will reduce the verified model from 1485 to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Omit the anchor row’s penultimate-pivot coordinate, reconstruct both omitted coordinates, and transfer the inverse shear into the corresponding key row.

EVIDENCE: The current normalized orthogonal-query design achieved 99.92% at 1485 parameters. The analogous target-first anchor shear reached only 91.15%, but prior first-head experiments showed strong directional asymmetry, making the complementary penultimate-first shear the closest distinct untested reduction.

<<<<<<< SEARCH
        self.q_anchor_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
=======
        self.q_anchor_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_anchor_relative = torch.cat(
            (
                self.q_anchor_weight[:q_anchor_pivot],
                self.q_anchor_weight.new_zeros(1),
                self.q_anchor_weight[q_anchor_pivot:],
            )
        )
=======
        q_anchor_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate
            not in (q_anchor_pivot, q_penultimate_pivot)
        ]
        q_anchor_relative = self.q_anchor_weight.new_zeros(
            d_model - 1
        )
        q_anchor_relative[q_anchor_free] = self.q_anchor_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
            q_anchor_pivot = int(
                q_shear_chart.abs().argmax().item()
            )
            anchor_shear = (
                q_anchor_free[q_anchor_pivot]
                / q_shear_chart[q_anchor_pivot]
            )
            q_anchor_chart = (
                q_anchor_free - anchor_shear * q_shear_chart
            )
            q_anchor_chart[q_anchor_pivot] = 0.0
=======
            anchor_penultimate_shear = (
                q_anchor_free[q_penultimate_pivot]
                / q_penultimate_relative[q_penultimate_pivot]
            )
            q_anchor_chart = (
                q_anchor_free
                - anchor_penultimate_shear
                * q_penultimate_relative
            )
            q_anchor_chart[q_penultimate_pivot] = 0.0

            q_anchor_pivot = int(
                q_shear_chart.abs().argmax().item()
            )
            anchor_shear = (
                q_anchor_chart[q_anchor_pivot]
                / q_shear_chart[q_anchor_pivot]
            )
            q_anchor_chart = (
                q_anchor_chart - anchor_shear * q_shear_chart
            )
            q_anchor_chart[q_anchor_pivot] = 0.0
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale * key_penultimate_sheared
                + bias_penultimate_shear * key_shear_free
            )
=======
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale * key_penultimate_sheared
                + bias_penultimate_shear * key_shear_free
                + anchor_penultimate_shear * key_anchor_free
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.q_anchor_weight = nn.Parameter(
                torch.cat(
                    (
                        q_anchor_chart[:q_anchor_pivot],
                        q_anchor_chart[q_anchor_pivot + 1:],
                    )
                )
            )
=======
            q_anchor_free_coordinates = [
                coordinate
                for coordinate in range(q_anchor_chart.numel())
                if coordinate
                not in (q_anchor_pivot, q_penultimate_pivot)
            ]
            block.attn.q_anchor_weight = nn.Parameter(
                q_anchor_chart[
                    q_anchor_free_coordinates
                ].clone()
            )
>>>>>>> REPLACE