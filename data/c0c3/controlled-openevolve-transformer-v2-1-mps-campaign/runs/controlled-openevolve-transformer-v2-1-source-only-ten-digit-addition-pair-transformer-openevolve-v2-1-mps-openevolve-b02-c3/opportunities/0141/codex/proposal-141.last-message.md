MECHANISM: Second-head Q/K coordinate-scale gauge fixing with vectorized QKV reconstruction

HYPOTHESIS: A 662-parameter transformer will achieve at least 99% accuracy because the second head’s remaining key-channel scale is an exact Q/K rescaling redundancy, while vectorized weight reconstruction addresses the prior attempt’s verification-time failure without changing model function.

INTENDED_EDIT: Reconstruct the qualified 663-parameter first-head-quintet design, fix the analogous remaining second-head key coefficient at 0.02, and replace iterative QKV assembly with an equivalent single scatter operation.

EVIDENCE: Fixing the first-head key-channel scale yielded 99.88% at 663 parameters; the analogous 662-parameter attempt timed out rather than producing negative accuracy evidence, while further relative-bias compression failed.

<<<<<<< SEARCH
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
        selected_indices = {
=======
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
        if head_dim > 2:
            selected_key_channels.update(
                head * head_dim + 2 for head in range(n_head)
            )
        selected_indices = {
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "fixed_coeff",
            torch.tensor(
                [
                    0.0 if index in shear_indices else 0.02
                    for index in self.fixed_indices
                ]
            ),
            persistent=False,
        )

        basis = torch.zeros(d_model, in_features)
=======
        fixed_indices = torch.tensor(self.fixed_indices)
        fixed_coeff = torch.tensor(
            [
                0.0 if index in shear_indices else 0.02
                for index in self.fixed_indices
            ]
        )
        fixed_weight = torch.zeros(out_features * in_features)
        fixed_weight[fixed_indices] = fixed_coeff
        learned_mask = torch.ones(
            out_features * in_features, dtype=torch.bool
        )
        learned_mask[fixed_indices] = False
        self.register_buffer(
            "fixed_weight", fixed_weight, persistent=False
        )
        self.register_buffer(
            "learned_indices",
            torch.arange(out_features * in_features)[learned_mask],
            persistent=False,
        )

        basis = torch.zeros(d_model, in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        learned_start = 0
        full_start = 0
        for fixed_index, fixed_coeff in zip(
            self.fixed_indices, self.fixed_coeff
        ):
            width = fixed_index - full_start
            pieces.append(
                self.coeff[learned_start : learned_start + width]
            )
            pieces.append(fixed_coeff.view(1))
            learned_start += width
            full_start = fixed_index + 1
        pieces.append(self.coeff[learned_start:])

        weight = torch.cat(pieces).view(
            self.out_features, self.in_features
        )
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fixed_weight.scatter(
            0, self.learned_indices, self.coeff
        ).view(self.out_features, self.in_features)
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent quartet immediately preceding them. The second
        # head fixes its next six longest biases, shares its three longest
        # remaining positions, and separately shares the adjacent pair
        # immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 15)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent quintet immediately preceding them. The second
        # head fixes its next six longest biases, shares its three longest
        # remaining positions, and separately shares the adjacent pair
        # immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 16)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 6
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 6 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias[:-1],
                first_head_bias[-1:].expand(4),
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 7
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 7 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias[:-1],
                first_head_bias[-1:].expand(5),
>>>>>>> REPLACE