MECHANISM: First-head residual value-subspace triangularization

HYPOTHESIS: Completing first-column value alignment in both heads, then eliminating two second-column coefficients within the first head’s preserved zero subspace, will produce a 1,609-parameter model with at least 99% accuracy.

INTENDED_EDIT: Generalize compensated value-basis rotations to arbitrary channel pairs and input columns; remove three first-column coefficients per head and two second-column coefficients in the first head.

EVIDENCE: The first residual-subspace reduction in head 0 achieved 99.73% at 1,610 parameters, while applying the next reduction to head 1 narrowly failed at 98.99%; using the remaining independent stabilizer in the already-qualified first-head zero subspace tests a distinct 1,609-parameter allocation.

<<<<<<< SEARCH
        # Fix two disjoint value-basis directions in every head when four
        # value channels are available, otherwise fix the single pair.
        value_basis_pairs = [(head, 0) for head in range(n_head)]
        if self.head_dim >= 4:
            value_basis_pairs.extend((head, 2) for head in range(n_head))
        self.value_basis_pairs = tuple(sorted(value_basis_pairs))
        self.value_fixed_indices = tuple(
            (2 * d_model + head * self.head_dim + local) * d_model
            for head, local in self.value_basis_pairs
        )
=======
        # Align each head's first value-weight column with one basis vector.
        # The resulting three-dimensional zero subspace in the first head can
        # then be triangularized through two second-column rotations.
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
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
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
            for head, local in self.value_basis_pairs:
                first_value = 2 * self.d_model + head * self.head_dim + local
                second_value = first_value + 1
                first_column = head * self.head_dim + local
                second_column = first_column + 1

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