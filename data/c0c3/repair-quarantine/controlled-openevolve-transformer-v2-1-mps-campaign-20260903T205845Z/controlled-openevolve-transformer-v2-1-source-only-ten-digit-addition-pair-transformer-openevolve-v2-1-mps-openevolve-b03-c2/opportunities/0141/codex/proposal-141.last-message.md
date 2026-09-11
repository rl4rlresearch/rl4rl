MECHANISM: Second-head bias-preserving anchor-to-target query/key shear

HYPOTHESIS: Fixing the anchor query row at the normalized target row’s stable pivot will reduce the model from 1485 to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Shear the second head’s anchor query row against its zero-bias normalized target row, omit the resulting fixed coordinate, and apply the inverse shear to the corresponding key row.

EVIDENCE: The current 1485-parameter design achieved 0.9992 accuracy and already uses the second head’s max-pivot target shear successfully; the failed 1484 attempt instead modified the first-head frame, motivating an independent reduction within the proven second-head construction.

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
        q_shear_relative = self.q_shear_weight.new_zeros(
=======
        q_anchor_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate
            not in (q_anchor_pivot, q_target_pivot)
        ]
        q_anchor_relative = self.q_anchor_weight.new_zeros(
            d_model - 1
        )
        q_anchor_relative[q_anchor_free] = self.q_anchor_weight
        q_shear_relative = self.q_shear_weight.new_zeros(
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
            anchor_target_shear = (
                q_anchor_free[q_target_pivot]
                / q_target_relative[q_target_pivot]
            )
            q_anchor_target_chart = (
                q_anchor_free
                - anchor_target_shear * q_target_relative
            )
            q_anchor_target_chart[q_target_pivot] = 0.0

            q_anchor_pivot = int(
                q_shear_chart.abs().argmax().item()
            )
            anchor_shear = (
                q_anchor_target_chart[q_anchor_pivot]
                / q_shear_chart[q_anchor_pivot]
            )
            q_anchor_chart = (
                q_anchor_target_chart
                - anchor_shear * q_shear_chart
            )
            q_anchor_chart[q_anchor_pivot] = 0.0
            q_anchor_chart[q_target_pivot] = 0.0
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + bias_target_shear * key_shear_free
            )
=======
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + bias_target_shear * key_shear_free
                + anchor_target_shear * key_anchor_free
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
            query_free = [
=======
            q_anchor_free_coordinates = [
                coordinate
                for coordinate in range(q_anchor_chart.numel())
                if coordinate
                not in (q_anchor_pivot, q_target_pivot)
            ]
            block.attn.q_anchor_weight = nn.Parameter(
                q_anchor_chart[q_anchor_free_coordinates].clone()
            )
            query_free = [
>>>>>>> REPLACE