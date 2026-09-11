MECHANISM: Residual value-subspace gauge fixing

HYPOTHESIS: Rotating two already first-column-zero value channels in the first attention head to eliminate one second-column coefficient will reduce the model from 1,611 to 1,610 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Generalize value-basis rotations to select their pivot input column, then add one compensated rotation within the first head’s three-dimensional zero subspace and omit the newly zero second-column coefficient.

EVIDENCE: The successive compensated value-basis rotations from 1,617 through 1,611 parameters all met the accuracy requirement, with the current symmetric cross-pair design reaching 99.75%; this tests the next unused exact orthogonal gauge direction without adding another sensitive positional constraint.

<<<<<<< SEARCH
        # Fix two disjoint value pairs in each four-dimensional head, then
        # rotate the two remaining nonzero first-column coordinates together.
        value_basis_rotations = []
        value_fixed_coordinates = []
        for head in range(n_head):
            value_basis_rotations.append((head, 0, 1))
            value_fixed_coordinates.append((head, 0))
            if self.head_dim >= 4:
                value_basis_rotations.append((head, 2, 3))
                value_fixed_coordinates.append((head, 2))
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
        # In the first head, two of the resulting zero coordinates retain an
        # orthogonal stabilizer that can also eliminate one second-column
        # coefficient without disturbing any first-column zero.
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