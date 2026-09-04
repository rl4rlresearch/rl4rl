MECHANISM: Affine query/key scale gauge on the dual-sheared second-head row

HYPOTHESIS: Normalizing the combined relative weights and learned bias of the already-successful dual-sheared second-head query row will reduce the qualified 1490-parameter construction to 1489 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified dual-target bias-preserving shears, then omit one additional query-weight coordinate by fixing an affine-row scale chart and absorbing its initialization scale into the matching key row.

EVIDENCE: The dual-target shear on this same second-head bias-bearing row achieved 99.91% at 1490 parameters, while changing the other bias-bearing row, the first head, or the value/output chart failed; an exact diagonal gauge on the successful row is the closest distinct reduction.

<<<<<<< SEARCH
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
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.register_buffer(
            "q_penultimate_pivot",
            torch.zeros((), dtype=torch.long),
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The second head's zero-bias query rows use their full
        # scale/shear gauge. Its neighboring affine query row is sheared
        # against both rows and then normalized with an exact query/key scale.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_shear_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.q_penultimate_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.q_target_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
        self.register_buffer(
            "q_shear_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_penultimate_pivot",
            torch.zeros((), dtype=torch.long),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        bias_split = self.head_dim - 2
        query_bias = torch.cat(
            (
                self.qkv.bias[:bias_split],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[bias_split:],
                self.qkv.bias.new_zeros(2),
            )
        )
        value_bias = self.qkv.bias.new_zeros(d_model)
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )

        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
        q_target_pivot = int(self.q_target_pivot.item())
        query_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate
            not in (q_penultimate_pivot, q_target_pivot)
        ]

        q_penultimate_chart = self.q_penultimate_weight.new_zeros(
            d_model - 1
        )
        q_penultimate_chart[q_penultimate_pivot] = 1.0
        q_penultimate_chart[query_free] = (
            self.q_penultimate_weight
        )
        q_penultimate_relative = q_penultimate_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_penultimate_chart.norm()
        )

        q_target_chart = self.q_target_weight.new_zeros(
            d_model - 1
        )
        q_target_chart[q_target_pivot] = 1.0
        q_target_chart[query_free] = self.q_target_weight
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
=======
        bsz, seqlen, d_model = x.shape
        q_shear_pivot = int(self.q_shear_pivot.item())
        q_penultimate_pivot = int(
            self.q_penultimate_pivot.item()
        )
        q_target_pivot = int(self.q_target_pivot.item())
        query_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate
            not in (q_penultimate_pivot, q_target_pivot)
        ]
        q_shear_free = [
            coordinate
            for coordinate in query_free
            if coordinate != q_shear_pivot
        ]

        q_shear_affine_chart = self.q_shear_weight.new_zeros(
            d_model
        )
        q_shear_affine_chart[q_shear_pivot] = 1.0
        q_shear_affine_chart[q_shear_free] = self.q_shear_weight
        q_shear_affine_chart[-1] = self.qkv.bias[-1]
        q_shear_affine = q_shear_affine_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_shear_affine_chart.norm()
        )
        q_shear_relative = q_shear_affine[:-1]

        bias_split = self.head_dim - 2
        query_bias_parameters = torch.cat(
            (self.qkv.bias[:-1], q_shear_affine[-1:])
        )
        query_bias = torch.cat(
            (
                query_bias_parameters[:bias_split],
                query_bias_parameters.new_zeros(2),
                query_bias_parameters[bias_split:],
                query_bias_parameters.new_zeros(2),
            )
        )
        value_bias = self.qkv.bias.new_zeros(d_model)
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )

        q_penultimate_chart = self.q_penultimate_weight.new_zeros(
            d_model - 1
        )
        q_penultimate_chart[q_penultimate_pivot] = 1.0
        q_penultimate_chart[query_free] = (
            self.q_penultimate_weight
        )
        q_penultimate_relative = q_penultimate_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_penultimate_chart.norm()
        )

        q_target_chart = self.q_target_weight.new_zeros(
            d_model - 1
        )
        q_target_chart[q_target_pivot] = 1.0
        q_target_chart[query_free] = self.q_target_weight
        q_target_relative = q_target_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_target_chart.norm()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            q_penultimate_scale = (
                q_penultimate_pivot_value.sign()
                * q_penultimate_sheared.norm()
                / q_penultimate_gauge_norm
            )

            q_target_pivot_value = q_target_sheared[q_target_pivot]
=======
            q_penultimate_scale = (
                q_penultimate_pivot_value.sign()
                * q_penultimate_sheared.norm()
                / q_penultimate_gauge_norm
            )
            q_penultimate_relative = q_penultimate_chart * (
                q_penultimate_gauge_norm
                / q_penultimate_chart.norm()
            )

            q_target_pivot_value = q_target_sheared[q_target_pivot]
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            bias_target_shear = (
                q_shear_free[q_target_pivot]
                / q_target_relative[q_target_pivot]
            )
            q_shear_chart = (
                q_shear_free
                - bias_target_shear * q_target_relative
            )
            q_shear_chart[q_target_pivot] = 0.0
            bias_penultimate_shear = (
                q_shear_chart[q_penultimate_pivot]
                / q_penultimate_relative[q_penultimate_pivot]
            )
            q_shear_chart = (
                q_shear_chart
                - bias_penultimate_shear
                * q_penultimate_relative
            )
            q_shear_chart[q_penultimate_pivot] = 0.0
            q_shear_chart[q_target_pivot] = 0.0

            q_shear_pivot = int(
                q_shear_chart.abs().argmax().item()
            )
            q_shear_pivot_value = q_shear_chart[q_shear_pivot]
            q_shear_normalized_chart = (
                q_shear_chart / q_shear_pivot_value
            )
            q_shear_gauge_norm = (
                0.02 * math.sqrt(q_shear_chart.numel())
            )
            q_shear_scale = (
                q_shear_pivot_value.sign()
                * q_shear_chart.norm()
                / q_shear_gauge_norm
            )

            key_shear = cfg.d_model + q_shear
>>>>>>> REPLACE

<<<<<<< SEARCH
            key_penultimate_free = full_qkv_weight[
                key_penultimate
            ].clone()
            key_target_free = full_qkv_weight[key_target].clone()
            key_target_sheared = (
                key_target_free
                + first_shear * key_penultimate_free
            )
            key_penultimate_sheared = (
                key_penultimate_free
                + second_shear * key_target_sheared
            )
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale * key_penultimate_sheared
            )
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + q_shear_coefficient
                * full_qkv_weight[key_shear]
            )
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
            )
=======
            key_shear_free = full_qkv_weight[key_shear].clone()
            key_penultimate_free = full_qkv_weight[
                key_penultimate
            ].clone()
            key_target_free = full_qkv_weight[key_target].clone()
            key_target_sheared = (
                key_target_free
                + first_shear * key_penultimate_free
            )
            key_penultimate_sheared = (
                key_penultimate_free
                + second_shear * key_target_sheared
            )
            full_qkv_weight[key_penultimate] = (
                q_penultimate_scale * key_penultimate_sheared
                + bias_penultimate_shear * key_shear_free
            )
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + bias_target_shear * key_shear_free
            )
            full_qkv_weight[key_shear] = (
                q_shear_scale * key_shear_free
            )
            block.attn.q_shear_pivot.fill_(q_shear_pivot)
            block.attn.q_penultimate_pivot.fill_(
                q_penultimate_pivot
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.q_shear_weight = nn.Parameter(
                torch.cat(
                    (
                        q_shear_chart[:q_target_pivot],
                        q_shear_chart[q_target_pivot + 1:],
                    )
                )
            )
            query_free = [
                coordinate
                for coordinate in range(q_target_chart.numel())
                if coordinate
                not in (q_penultimate_pivot, q_target_pivot)
            ]
=======
            query_free = [
                coordinate
                for coordinate in range(q_target_chart.numel())
                if coordinate
                not in (q_penultimate_pivot, q_target_pivot)
            ]
            q_shear_free = [
                coordinate
                for coordinate in query_free
                if coordinate != q_shear_pivot
            ]
            block.attn.q_shear_weight = nn.Parameter(
                q_shear_normalized_chart[q_shear_free].clone()
            )
>>>>>>> REPLACE