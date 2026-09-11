MECHANISM: Bias-free diagonal query/key scale chart on the mutually sheared anchor row

HYPOTHESIS: Normalizing only the relative query weights of the newly qualified mutually sheared anchor row, while leaving its query bias independently learned, will reduce the model to 1488 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Add a max-pivot scale chart for the second head’s anchor query row, omit one learned weight coordinate, and transfer the initialization scale into the matching key row.

EVIDENCE: The current mutual bias-bearing shear reached 99.99% accuracy at 1489 parameters, and diagonal query/key scale charts already succeed on both zero-bias target rows. Unlike the failed affine chart, this construction does not normalize or couple the learned query bias.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The second head's zero-bias query rows use their full
        # scale/shear gauge. One bias-bearing row is sheared against both,
        # and the other is sheared against that freely biased row.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_anchor_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The second head's zero-bias query rows use their full
        # scale/shear gauge. One bias-bearing row is sheared against both;
        # the other is sheared against it and uses a weight-only scale gauge
        # while retaining its independently learned bias.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_anchor_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "q_anchor_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_penultimate_pivot",
=======
        self.register_buffer(
            "q_anchor_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_anchor_scale_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_penultimate_pivot",
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_anchor_pivot = int(self.q_anchor_pivot.item())
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
=======
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_anchor_scale_pivot = int(
            self.q_anchor_scale_pivot.item()
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
        anchor_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate
            not in (q_anchor_pivot, q_anchor_scale_pivot)
        ]
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
        q_anchor_chart = self.q_anchor_weight.new_zeros(
            d_model - 1
        )
        q_anchor_chart[q_anchor_scale_pivot] = 1.0
        q_anchor_chart[anchor_free] = self.q_anchor_weight
        q_anchor_relative = q_anchor_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_anchor_chart.norm()
        )
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
            q_anchor_scale_pivot = int(
                q_anchor_chart.abs().argmax().item()
            )
            q_anchor_scale_pivot_value = q_anchor_chart[
                q_anchor_scale_pivot
            ]
            q_anchor_scale_chart = (
                q_anchor_chart / q_anchor_scale_pivot_value
            )
            q_anchor_gauge_norm = (
                0.02 * math.sqrt(q_anchor_chart.numel())
            )
            q_anchor_scale = (
                q_anchor_scale_pivot_value.sign()
                * q_anchor_chart.norm()
                / q_anchor_gauge_norm
            )

            key_anchor = cfg.d_model + q_anchor
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_qkv_weight[key_shear] = (
                key_shear_free + anchor_shear * key_anchor_free
            )
            block.attn.q_anchor_pivot.fill_(q_anchor_pivot)
            block.attn.q_penultimate_pivot.fill_(
=======
            full_qkv_weight[key_anchor] = (
                q_anchor_scale * key_anchor_free
            )
            full_qkv_weight[key_shear] = (
                key_shear_free + anchor_shear * key_anchor_free
            )
            block.attn.q_anchor_pivot.fill_(q_anchor_pivot)
            block.attn.q_anchor_scale_pivot.fill_(
                q_anchor_scale_pivot
            )
            block.attn.q_penultimate_pivot.fill_(
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
            q_anchor_coordinates = [
                coordinate
                for coordinate in range(
                    q_anchor_scale_chart.numel()
                )
                if coordinate
                not in (q_anchor_pivot, q_anchor_scale_pivot)
            ]
            block.attn.q_anchor_weight = nn.Parameter(
                q_anchor_scale_chart[
                    q_anchor_coordinates
                ].clone()
            )
            query_free = [
>>>>>>> REPLACE