MECHANISM: Augmented-affine biased-query orthogonality gauge

HYPOTHESIS: Projecting the first head’s second biased query row orthogonally to the first in joint weight-and-bias space will reduce the verified model from 1485 to 1484 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace one biased-query weight coordinate with a stable projection chart that preserves both learned biases, and absorb the initialization shear into the corresponding key row.

EVIDENCE: The weight-only biased-pair constraint reached 80.47% and tying the biases reached 70.49%, indicating that preserving the affine bias degree of freedom is important; the current unconstrained affine design reaches 99.92%.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_first_weight = nn.Parameter(
            torch.empty(d_model - 3)
        )
=======
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.q_biased_weight = nn.Parameter(
            torch.empty(d_model - 2)
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
=======
        bsz, seqlen, d_model = x.shape
        bias_split = self.head_dim - 2
        q_biased_pivot = int(self.q_biased_pivot.item())
        q_biased_free = [
            coordinate
            for coordinate in range(d_model - 1)
            if coordinate != q_biased_pivot
        ]
        q_biased_chart_relative = self.q_biased_weight.new_zeros(
            d_model - 1
        )
        q_biased_chart_relative[q_biased_free] = (
            self.q_biased_weight
        )
        q_biased_chart = torch.cat(
            (
                q_biased_chart_relative,
                self.qkv.bias[1:2],
            )
        )
        q_biased_first_augmented = torch.cat(
            (
                self.qkv.weight[0],
                self.qkv.bias[:1],
            )
        )
        q_biased_augmented = q_biased_chart - (
            (q_biased_chart * q_biased_first_augmented).sum()
            / q_biased_first_augmented.square().sum()
        ) * q_biased_first_augmented
        q_biased_relative = q_biased_augmented[:-1]
        q_biased_bias = q_biased_augmented[-1:]

        query_bias = torch.cat(
            (
                self.qkv.bias[:1],
                q_biased_bias,
                self.qkv.bias[2:bias_split],
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
        q_biased_row = 1
        q_first_neighbor = self.head_dim - 2
        stored_first_neighbor = q_first_neighbor - 1
        qkv_rows = torch.cat(
            (
                self.qkv.weight[:q_biased_row],
                q_biased_relative.unsqueeze(0),
                self.qkv.weight[
                    q_biased_row:stored_first_neighbor
                ],
                q_neighbor_relative.unsqueeze(0),
                q_first_relative.unsqueeze(0),
                q_anchor_relative.unsqueeze(0),
                q_shear_relative.unsqueeze(0),
                q_penultimate_relative.unsqueeze(0),
                q_target_relative.unsqueeze(0),
                self.qkv.weight[stored_first_neighbor:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_first_neighbor = block.attn.head_dim - 2
=======
            relative_qkv_weight = (
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            q_biased_first = 0
            q_biased_target = 1
            qkv_bias_chart = block.attn.qkv.bias.detach().clone()
            q_biased_first_augmented = torch.cat(
                (
                    relative_qkv_weight[q_biased_first],
                    qkv_bias_chart[q_biased_first:q_biased_target],
                )
            )
            q_biased_target_augmented = torch.cat(
                (
                    relative_qkv_weight[q_biased_target],
                    qkv_bias_chart[
                        q_biased_target:q_biased_target + 1
                    ],
                )
            )
            q_biased_shear = (
                q_biased_target_augmented
                * q_biased_first_augmented
            ).sum() / q_biased_first_augmented.square().sum()
            q_biased_orthogonal = (
                q_biased_target_augmented
                - q_biased_shear * q_biased_first_augmented
            )
            q_biased_pivot = int(
                q_biased_first_augmented[:-1].abs().argmax().item()
            )
            q_biased_chart_shear = (
                q_biased_orthogonal[q_biased_pivot]
                / q_biased_first_augmented[q_biased_pivot]
            )
            q_biased_chart = (
                q_biased_orthogonal
                - q_biased_chart_shear
                * q_biased_first_augmented
            )
            q_biased_chart[q_biased_pivot] = 0.0
            key_biased_first = cfg.d_model + q_biased_first
            key_biased_target = cfg.d_model + q_biased_target
            key_biased_target_free = full_qkv_weight[
                key_biased_target
            ].clone()
            full_qkv_weight[key_biased_first] = (
                full_qkv_weight[key_biased_first]
                + q_biased_shear * key_biased_target_free
            )
            qkv_bias_chart[q_biased_target] = q_biased_chart[-1]
            block.attn.qkv.bias = nn.Parameter(qkv_bias_chart)
            block.attn.q_biased_pivot.fill_(q_biased_pivot)
            q_biased_free_coordinates = [
                coordinate
                for coordinate in range(d_model - 1)
                if coordinate != q_biased_pivot
            ]

            q_first_neighbor = block.attn.head_dim - 2
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
                        relative_qkv_weight[:q_biased_target],
                        relative_qkv_weight[
                            q_biased_target + 1:q_first_neighbor
                        ],
                        relative_qkv_weight[q_target + 1:],
                    ),
                    dim=0,
                )
            )
            block.attn.q_biased_weight = nn.Parameter(
                q_biased_chart[
                    q_biased_free_coordinates
                ].clone()
            )
            q_first_free_coordinates = [
>>>>>>> REPLACE