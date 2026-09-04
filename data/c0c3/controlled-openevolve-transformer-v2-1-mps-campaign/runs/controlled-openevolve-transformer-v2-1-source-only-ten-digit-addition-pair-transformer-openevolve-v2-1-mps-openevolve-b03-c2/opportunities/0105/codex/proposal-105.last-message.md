MECHANISM: First-head joint affine query/key scale quotient

HYPOTHESIS: Normalizing one first-head biased query in joint relative-weight-and-bias space and transferring its scale to the matching key will reduce the verified model from 1485 to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store the first biased query as a max-pivot fixed-norm affine chart, remove its bias from `qkv.bias`, omit its weight row from `qkv.weight`, and absorb its initialization scale into the paired key row.

EVIDENCE: The current normalized-frame model achieved 99.92% at 1485 parameters. Tested 1484 changes that constrained weight geometry or tied biases failed, while augmented-affine orthogonality reached 92.63%; this tests the less restrictive exact scale redundancy while preserving both biased-query directions and independent biases.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The first head's zero-bias query pair forms a
        # normalized orthogonal frame. The second head's zero-bias rows use
        # their scale/shear gauge, one biased row is sheared against both,
        # and the other biased row is sheared against that freely biased row.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. One first-head biased query uses a joint affine scale
        # quotient, while its zero-bias pair forms a normalized orthogonal
        # frame. The second head's zero-bias rows use their scale/shear gauge,
        # one biased row is sheared against both, and the other biased row is
        # sheared against that freely biased row.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 5))
        self.q_biased_weight = nn.Parameter(
            torch.empty(d_model - 1)
        )
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "q_first_pivot",
            torch.zeros((), dtype=torch.long),
        )
=======
        self.register_buffer(
            "q_biased_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "q_first_pivot",
            torch.zeros((), dtype=torch.long),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias_split = self.head_dim - 2
        query_bias = torch.cat(
            (
                self.qkv.bias[:bias_split],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[bias_split:],
                self.qkv.bias.new_zeros(2),
            )
        )
=======
        bias_split = self.head_dim - 2
        q_biased_pivot = int(self.q_biased_pivot.item())
        q_biased_free = [
            coordinate
            for coordinate in range(d_model)
            if coordinate != q_biased_pivot
        ]
        q_biased_chart = self.q_biased_weight.new_zeros(
            d_model
        )
        q_biased_chart[q_biased_pivot] = 1.0
        q_biased_chart[q_biased_free] = self.q_biased_weight
        q_biased_affine = q_biased_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / q_biased_chart.norm()
        )
        q_biased_relative = q_biased_affine[:-1]
        q_biased_bias = q_biased_affine[-1:]
        query_bias = torch.cat(
            (
                q_biased_bias,
                self.qkv.bias[:bias_split - 1],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[bias_split - 1:],
                self.qkv.bias.new_zeros(2),
            )
        )
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
            block.attn.q_neighbor_pivot.fill_(q_neighbor_pivot)

            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_anchor = cfg.d_model - 4
=======
            block.attn.q_neighbor_pivot.fill_(q_neighbor_pivot)

            q_biased_weight_part = relative_qkv_weight[0]
            q_biased_affine = torch.cat(
                (
                    q_biased_weight_part,
                    q_biased_weight_part.new_zeros(1),
                )
            )
            q_biased_pivot = int(
                q_biased_weight_part.abs().argmax().item()
            )
            q_biased_pivot_value = q_biased_affine[
                q_biased_pivot
            ]
            q_biased_chart = (
                q_biased_affine / q_biased_pivot_value
            )
            q_biased_gauge_norm = (
                0.02 * math.sqrt(q_biased_weight_part.numel())
            )
            q_biased_scale = (
                q_biased_pivot_value.sign()
                * q_biased_affine.norm()
                / q_biased_gauge_norm
            )
            key_biased = cfg.d_model
            full_qkv_weight[key_biased] = (
                q_biased_scale * full_qkv_weight[key_biased]
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
            q_first_free_coordinates = [
=======
            block.attn.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_qkv_weight[1:q_first_neighbor],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            q_biased_free_coordinates = [
                coordinate
                for coordinate in range(q_biased_chart.numel())
                if coordinate != q_biased_pivot
            ]
            block.attn.q_biased_weight = nn.Parameter(
                q_biased_chart[
                    q_biased_free_coordinates
                ].clone()
            )
            q_first_free_coordinates = [
>>>>>>> REPLACE