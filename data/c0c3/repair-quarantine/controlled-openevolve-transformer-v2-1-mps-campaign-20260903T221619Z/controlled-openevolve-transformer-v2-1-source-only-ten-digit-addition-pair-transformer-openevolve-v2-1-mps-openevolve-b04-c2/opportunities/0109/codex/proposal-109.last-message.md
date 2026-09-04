MECHANISM: Head-0 terminal value-basis triangularization

HYPOTHESIS: Fixing the final head-0 value-gauge coordinate will produce a 1,575-parameter model with at least 99% accuracy.

INTENDED_EDIT: Add a third-column rotation between the two head-0 value rows already zero on columns 0 and 1, then omit the resulting zero coefficient.

EVIDENCE: The current nine-rotation model achieved 99.99% at 1,576 parameters, while adding the tenth rotation in head 1 reached only 98.58%. Head 0 already tolerates deeper triangularization, motivating testing its remaining exact gauge direction instead.

<<<<<<< SEARCH
        # Align each head's first value-weight column with one basis vector,
        # then triangularize their remaining zero subspaces through selected
        # second-column rotations.
=======
        # Align each head's first value-weight column with one basis vector,
        # then triangularize the remaining zero subspaces through selected
        # second- and third-column rotations.
>>>>>>> REPLACE

<<<<<<< SEARCH
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
                elif head == 1:
=======
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
                    value_basis_rotations.append((head, 0, 1, 2))
                    value_fixed_coordinates.append((head, 0, 2))
                elif head == 1:
>>>>>>> REPLACE