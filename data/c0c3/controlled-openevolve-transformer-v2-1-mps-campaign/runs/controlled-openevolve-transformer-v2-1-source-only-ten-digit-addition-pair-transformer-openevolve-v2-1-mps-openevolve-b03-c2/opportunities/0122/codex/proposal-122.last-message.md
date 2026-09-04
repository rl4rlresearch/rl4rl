MECHANISM: Homogeneous affine query/key scale gauge with final-only validation

HYPOTHESIS: Normalizing the second-head anchor’s complete affine query row and transferring its scale into the matching key row will reduce the qualified 1484-parameter model to 1483 parameters while retaining at least 99% accuracy and completing at step 4999.

INTENDED_EDIT: Store the anchor’s relative weights and bias in one fixed-norm chart, absorb the omitted scale into its key row, remove the corresponding standalone query-bias parameter, and skip intermediate validation.

EVIDENCE: The current design achieved 99.8% at 1484 parameters, while the prior affine-anchor normalization trial timed out without negative accuracy evidence; this retries that expressivity-preserving gauge with reduced validation overhead.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head's zero-bias query pair forms a
        # normalized orthogonal frame. The second head's zero-bias rows use
        # their scale/shear gauge, one biased row is sheared against both,
        # and the other biased row is sheared against that freely biased row.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head's zero-bias query pair forms a
        # normalized orthogonal frame. The second head's zero-bias rows use
        # their scale/shear gauge, while the anchor's complete affine row is
        # normalized and its omitted scale is absorbed into the matching key.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 5))
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        q_first_pivot = int(self.q_first_pivot.item())
        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_neighbor_pivot = int(self.q_neighbor_pivot.item())
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_penultimate_pivot = int(
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_anchor_pivot = int(self.q_anchor_pivot.item())
        q_anchor_scale_pivot = int(
            self.q_anchor_scale_pivot.item()
        )
        q_anchor_affine_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate != q_anchor_scale_pivot
        ]
        q_anchor_affine_chart = self.q_anchor_weight.new_zeros(
            d_model - 1
        )
        q_anchor_affine_chart[q_anchor_scale_pivot] = 1.0
        q_anchor_affine_chart[q_anchor_affine_free] = (
            self.q_anchor_weight
        )
        q_anchor_affine = q_anchor_affine_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_anchor_affine_chart.norm()
        )
        q_anchor_weight_coordinates = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate != q_anchor_pivot
        ]
        q_anchor_relative = self.q_anchor_weight.new_zeros(
            d_model - 1
        )
        q_anchor_relative[q_anchor_weight_coordinates] = (
            q_anchor_affine[:-1]
        )
        q_anchor_bias = q_anchor_affine[-1:]

        bias_split = self.head_dim - 2
        query_bias = torch.cat(
            (
                self.qkv.bias[:bias_split],
                self.qkv.bias.new_zeros(2),
                q_anchor_bias,
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

        q_first_pivot = int(self.q_first_pivot.item())
        q_first_shear_pivot = int(
            self.q_first_shear_pivot.item()
        )
        q_neighbor_pivot = int(self.q_neighbor_pivot.item())
        q_penultimate_pivot = int(
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
        q_shear_relative = self.q_shear_weight.new_zeros(
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
            q_anchor_weight_coordinates = [
                coordinate
                for coordinate in range(q_anchor_chart.numel())
                if coordinate != q_anchor_pivot
            ]
            q_anchor_affine = torch.cat(
                (
                    q_anchor_chart[q_anchor_weight_coordinates],
                    q_anchor_chart.new_zeros(1),
                )
            )
            q_anchor_scale_pivot = int(
                q_anchor_affine.abs().argmax().item()
            )
            q_anchor_scale_pivot_value = q_anchor_affine[
                q_anchor_scale_pivot
            ]
            q_anchor_scale_chart = (
                q_anchor_affine / q_anchor_scale_pivot_value
            )
            q_anchor_gauge_norm = (
                0.02 * math.sqrt(q_anchor_affine.numel())
            )
            q_anchor_scale = (
                q_anchor_scale_pivot_value.sign()
                * q_anchor_affine.norm()
                / q_anchor_gauge_norm
            )

            key_anchor = cfg.d_model + q_anchor
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + bias_target_shear * key_shear_free
            )
            full_qkv_weight[key_shear] = (
                key_shear_free + anchor_shear * key_anchor_free
            )
            block.attn.q_anchor_pivot.fill_(q_anchor_pivot)
            block.attn.q_penultimate_pivot.fill_(
=======
            full_qkv_weight[key_target] = (
                q_target_scale * key_target_sheared
                + bias_target_shear * key_shear_free
            )
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
=======
            q_anchor_affine_free = [
                coordinate
                for coordinate in range(q_anchor_scale_chart.numel())
                if coordinate != q_anchor_scale_pivot
            ]
            block.attn.q_anchor_weight = nn.Parameter(
                q_anchor_scale_chart[
                    q_anchor_affine_free
                ].clone()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE