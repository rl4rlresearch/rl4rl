MECHANISM: Nested residual value-subspace triangularization

HYPOTHESIS: Eliminating a third-column coefficient within head 0’s two-dimensional subspace that already vanishes on the first two input columns will reduce the model from 1,609 to 1,608 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Add one compensated value-basis rotation between head 0 channels 0 and 1 at input column 2, then omit the coefficient made exactly zero.

EVIDENCE: The current design reached 99.87% at 1,609 parameters by placing both residual second-column reductions in head 0, while distributing the second reduction to head 1 scored 98.99%; this extends the successful head-0 triangularization through its remaining exact stabilizer.

<<<<<<< SEARCH
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
=======
        # Align each head's first value-weight column with one basis vector.
        # In the first head, triangularize the resulting zero subspace across
        # the second column, then use its remaining stabilizer on column three.
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
                    value_basis_rotations.append((head, 0, 1, 2))
                    value_fixed_coordinates.append((head, 0, 2))
>>>>>>> REPLACE