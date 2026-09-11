MECHANISM: Symmetric residual value-subspace gauge fixing

HYPOTHESIS: Applying the qualified 1,610-parameter design’s second-column value rotation to the corresponding zero subspace in the second attention head will yield 1,609 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Generalize rotations to select an input-column pivot, complete the verified first-column gauge fixes in both heads, and eliminate one second-column coefficient per head with compensated output-projection rotations.

EVIDENCE: The first residual-subspace rotation achieved 99.73% accuracy at 1,610 parameters; the identical unused gauge direction remains in the second head, and every value-basis reduction from 1,617 through 1,610 met the requirement.

<<<<<<< SEARCH
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
=======
        # Align each head's first value-weight column with one basis vector.
        # The resulting three-dimensional zero subspace can then eliminate
        # one second-column coefficient without disturbing those fixed zeros.
        value_basis_rotations = []
        value_fixed_coordinates = []
        for head in range(n_head):
            value_basis_rotations.append((head, 0, 1, 0))
            value_fixed_coordinates.append((head, 0, 0))
            if self.head_dim >= 4:
                value_basis_rotations.append((head, 2, 3, 0))
                value_fixed_coordinates.append((head, 2, 0))
                value_basis_rotations.append((head, 1, 3, 0))
                value_fixed_coordinates.append((head, 1, 0))
                value_basis_rotations.append((head, 0, 1, 1))
                value_fixed_coordinates.append((head, 0, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
        self.value_fixed_indices = tuple(
            sorted(
                (2 * d_model + head * self.head_dim + local) * d_model
                + input_column
                for head, local, input_column in value_fixed_coordinates
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            for head, first_local, second_local in self.value_basis_rotations:
                first_value = (
                    2 * self.d_model + head * self.head_dim + first_local
                )
                second_value = (
                    2 * self.d_model + head * self.head_dim + second_local
                )
                first_column = head * self.head_dim + first_local
                second_column = head * self.head_dim + second_local

                a = qkv_weight[first_value, 0]
                b = qkv_weight[second_value, 0]
=======
            for (
                head,
                first_local,
                second_local,
                input_column,
            ) in self.value_basis_rotations:
                first_value = (
                    2 * self.d_model + head * self.head_dim + first_local
                )
                second_value = (
                    2 * self.d_model + head * self.head_dim + second_local
                )
                first_column = head * self.head_dim + first_local
                second_column = head * self.head_dim + second_local

                a = qkv_weight[first_value, input_column]
                b = qkv_weight[second_value, input_column]
>>>>>>> REPLACE