MECHANISM: Head-local value-basis gauge fixation

HYPOTHESIS: Adding the qualified head-1 second-column rotation will reduce the model from 1,577 to 1,576 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Omit one additional value-weight coefficient through a head-1 rotation with the matching output-projection counter-rotation.

EVIDENCE: The identical ninth value-coordinate rotation achieved 99.99% accuracy with 1,576 parameters; further triangularization failed, so this applies only the verified reduction.

<<<<<<< SEARCH
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
=======
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
                elif head == 1:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
>>>>>>> REPLACE