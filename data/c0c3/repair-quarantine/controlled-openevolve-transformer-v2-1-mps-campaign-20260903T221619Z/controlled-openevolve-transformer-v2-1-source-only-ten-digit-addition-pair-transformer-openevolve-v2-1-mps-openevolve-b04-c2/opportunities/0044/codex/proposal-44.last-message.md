MECHANISM: Cross-pair intra-head value-basis gauge fixing

HYPOTHESIS: Extending the qualified 1,613-parameter design by rotating the two remaining nonzero first-column value coordinates in the first head will yield 1,612 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce both disjoint value-pair fixes in every head, then apply one compensated cross-pair rotation in the first head and omit the additional coefficient made exactly zero.

EVIDENCE: Complete disjoint value-basis gauge fixing achieved 99.85% at 1,613 parameters; the added rotation uses the same successful exact symmetry while preserving the previously fixed zeros.

<<<<<<< SEARCH
        self.d_model = d_model
        self.value_fixed_indices = tuple(
            (2 * d_model + head * self.head_dim) * d_model
            for head in range(n_head)
        )
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.d_model = d_model

        # First fix two disjoint pairs in every four-dimensional head. In the
        # first head, the remaining two nonzero first-column coordinates can
        # then be rotated together without disturbing either existing zero.
        value_basis_rotations = []
        value_fixed_coordinates = []
        for head in range(n_head):
            value_basis_rotations.append((head, 0, 1))
            value_fixed_coordinates.append((head, 0))
            if self.head_dim >= 4:
                value_basis_rotations.append((head, 2, 3))
                value_fixed_coordinates.append((head, 2))
                if head == 0:
                    value_basis_rotations.append((head, 1, 3))
                    value_fixed_coordinates.append((head, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
        self.value_fixed_indices = tuple(
            sorted(
                (2 * d_model + head * self.head_dim + local) * d_model
                for head, local in value_fixed_coordinates
            )
        )
        self.qkv = nn.Linear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Attention mixes value coordinates identically within each head.
        # Rotate two coordinates per head, counter-rotate the corresponding
        # output columns, and gauge-fix one coefficient in each rotation.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

            for head in range(self.n_head):
                first_value = 2 * self.d_model + head * self.head_dim
                second_value = first_value + 1
                first_column = head * self.head_dim
                second_column = first_column + 1
=======
        # Attention mixes value coordinates identically within each head.
        # Rotate each selected pair, counter-rotate the corresponding output
        # columns, and omit the coefficient made exactly zero.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

            for head, first_local, second_local in self.value_basis_rotations:
                first_value = (
                    2 * self.d_model + head * self.head_dim + first_local
                )
                second_value = (
                    2 * self.d_model + head * self.head_dim + second_local
                )
                first_column = head * self.head_dim + first_local
                second_column = head * self.head_dim + second_local
>>>>>>> REPLACE