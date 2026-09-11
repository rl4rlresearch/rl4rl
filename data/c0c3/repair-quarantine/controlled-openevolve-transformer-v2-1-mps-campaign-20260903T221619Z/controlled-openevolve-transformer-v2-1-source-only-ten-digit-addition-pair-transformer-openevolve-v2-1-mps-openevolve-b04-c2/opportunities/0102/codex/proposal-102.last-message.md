MECHANISM: Joint query–key orthogonal gauge fixation

HYPOTHESIS: Adding one joint query–key rotation to the verified 1,577-parameter row-12 design will produce a 1,576-parameter model with at least 99% accuracy.

INTENDED_EDIT: Compact positional row 12 as previously verified, then jointly rotate one query/key coordinate pair and omit the query coefficient made exactly zero.

EVIDENCE: The row-12 positional design achieved 99.78% at 1,577 parameters; the qualified models also successfully omit eight coefficients through analogous value-coordinate rotations, while further positional compaction at row 13 failed and row-14 attempts were unverifiable.

<<<<<<< SEARCH
        # Align each head's first value-weight column with one basis vector.
        # The resulting zero subspace in the first head is further
        # triangularized through two second-column rotations.
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
=======
        # Jointly rotating query and key coordinates preserves their dot
        # products, so align one query coefficient with that exact gauge.
        qk_basis_rotations = [(0, 0, 1, 0)]
        qk_fixed_coordinates = [(0, 0, 0)]

        # Align each head's first value-weight column with one basis vector.
        # The resulting zero subspace in the first head is further
        # triangularized through two second-column rotations.
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
        self.qk_basis_rotations = tuple(qk_basis_rotations)
        self.value_basis_rotations = tuple(value_basis_rotations)
        fixed_indices = [
            (head * self.head_dim + local) * d_model + input_column
            for head, local, input_column in qk_fixed_coordinates
        ]
        fixed_indices.extend(
            (2 * d_model + head * self.head_dim + local) * d_model
            + input_column
            for head, local, input_column in value_fixed_coordinates
        )
        self.value_fixed_indices = tuple(sorted(fixed_indices))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def compact_value_basis(self) -> None:
        # Rotate value coordinates and counter-rotate their output columns.
        # Also quotient every projection column except optimization-sensitive
        # column three.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

=======
    def compact_value_basis(self) -> None:
        # Rotate query and key coordinates together, then rotate value
        # coordinates while counter-rotating their output columns. Also
        # quotient every projection column except sensitive column three.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

            for (
                head,
                first_local,
                second_local,
                input_column,
            ) in self.qk_basis_rotations:
                first_query = head * self.head_dim + first_local
                second_query = head * self.head_dim + second_local
                a = qkv_weight[first_query, input_column]
                b = qkv_weight[second_query, input_column]
                norm = torch.hypot(a, b)
                cosine = b / norm
                sine = a / norm

                for offset in (0, self.d_model):
                    first_row = offset + head * self.head_dim + first_local
                    second_row = offset + head * self.head_dim + second_local
                    row0 = qkv_weight[first_row].clone()
                    row1 = qkv_weight[second_row].clone()
                    qkv_weight[first_row] = cosine * row0 - sine * row1
                    qkv_weight[second_row] = sine * row0 + cosine * row1

>>>>>>> REPLACE

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-11,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 20
=======
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 21
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (
                    initialized_pos[:7],
                    initialized_pos[8:12],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[12:-9]), dim=0
            )
=======
            selected_pos = torch.cat(
                (
                    initialized_pos[:7],
                    initialized_pos[8:13],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[13:-9]), dim=0
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                unrestricted_rows[:1],
                compact_rows[7:11],
                unrestricted_rows[1:],
                compact_rows[11:],
            ),
            dim=0,
        )
=======
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                unrestricted_rows[:1],
                compact_rows[7:12],
                unrestricted_rows[1:],
                compact_rows[12:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:11].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[11:].reshape(-1),
                    )
                )
=======
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:12].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[12:].reshape(-1),
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (
                        full_update[:7],
                        full_update[8:12],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[12:-9]), dim=0
                )
=======
                selected_update = torch.cat(
                    (
                        full_update[:7],
                        full_update[8:13],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[13:-9]), dim=0
                )
>>>>>>> REPLACE