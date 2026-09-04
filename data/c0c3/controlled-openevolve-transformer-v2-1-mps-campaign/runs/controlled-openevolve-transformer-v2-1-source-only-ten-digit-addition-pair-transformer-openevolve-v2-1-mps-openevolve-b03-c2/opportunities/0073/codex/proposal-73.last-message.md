MECHANISM: Bias-preserving second-head query/key shear

HYPOTHESIS: Shearing the second head’s last bias-bearing query row against its normalized zero-bias target row will reduce the model from 1492 to 1491 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Omit the bias-bearing query row’s target-pivot weight coordinate, reconstruct it as zero, and apply the inverse shear to the matching target key row at initialization.

EVIDENCE: The two zero-bias shears reached 99.98% at 1492 parameters, while deleting another second-head query bias collapsed to 47.99%; this extends the successful second-head shear gauge without restricting its learned bias.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final two query-bias
        # coordinates in each head. The second head's trailing query rows use
        # two scale gauges and both off-diagonal query/key shear gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final two query-bias
        # coordinates in each head. The second head's trailing query rows use
        # their full scale/shear gauge, and its neighboring bias-bearing row
        # uses an additional bias-preserving shear against the target row.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_target_relative = q_target_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_target_chart.norm()
        )

        q_penultimate_row = d_model - 2
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_penultimate_row],
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_penultimate_row:],
            ),
            dim=0,
        )
=======
        q_target_relative = q_target_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_target_chart.norm()
        )

        q_shear_relative = torch.cat(
            (
                self.q_shear_weight[:q_target_pivot],
                self.q_shear_weight.new_zeros(1),
                self.q_shear_weight[q_target_pivot:],
            )
        )
        q_shear_row = d_model - 3
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_shear_row],
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[q_shear_row:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
            q_penultimate_free = relative_qkv_weight[q_penultimate]
            q_target_free = relative_qkv_weight[q_target]
=======
            q_shear = cfg.d_model - 3
            q_penultimate = cfg.d_model - 2
            q_target = cfg.d_model - 1
            q_shear_free = relative_qkv_weight[q_shear]
            q_penultimate_free = relative_qkv_weight[q_penultimate]
            q_target_free = relative_qkv_weight[q_target]
>>>>>>> REPLACE

<<<<<<< SEARCH
            q_target_scale = (
                q_target_pivot_value.sign()
                * q_target_sheared.norm()
                / q_target_gauge_norm
            )

            key_penultimate = cfg.d_model + q_penultimate
=======
            q_target_scale = (
                q_target_pivot_value.sign()
                * q_target_sheared.norm()
                / q_target_gauge_norm
            )
            q_target_relative = q_target_chart * (
                q_target_gauge_norm / q_target_chart.norm()
            )
            q_shear_coefficient = (
                q_shear_free[q_target_pivot]
                / q_target_relative[q_target_pivot]
            )
            q_shear_chart = (
                q_shear_free
                - q_shear_coefficient * q_target_relative
            )
            q_shear_chart[q_target_pivot] = 0.0

            key_shear = cfg.d_model + q_shear
            key_penultimate = cfg.d_model + q_penultimate
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
            )
=======
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + q_shear_coefficient
                * full_qkv_weight[key_shear]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_penultimate],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            query_free = [
=======
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[:q_shear],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_shear_weight = nn.Parameter(
                torch.cat(
                    (
                        q_shear_chart[:q_target_pivot],
                        q_shear_chart[q_target_pivot + 1:],
                    )
                )
            )
            query_free = [
>>>>>>> REPLACE