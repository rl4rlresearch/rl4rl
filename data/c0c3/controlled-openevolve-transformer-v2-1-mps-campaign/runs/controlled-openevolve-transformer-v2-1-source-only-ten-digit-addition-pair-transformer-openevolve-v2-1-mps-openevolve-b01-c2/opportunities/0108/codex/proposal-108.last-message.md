MECHANISM: Second orthonormal token–position translation quotient

HYPOTHESIS: Constraining position zero to the Helmert subspace orthogonal to both its common direction and the coordinate-2-versus-4 contrast will reduce the verified model from 1,511 to 1,510 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove the first Helmert coordinate from position zero, transfer that omitted initialization component into the tied token embedding, and retain the existing dense four-coordinate AdamW update.

EVIDENCE: The 1,511-parameter position-zero Helmert design achieved 99.46%, and a Helmert chart previously made query row 6 viable after its anchored-coordinate form failed. The proposed 1,510-parameter positional contrast experiment was not tested because its patch matched multiple locations.

<<<<<<< SEARCH
        self.position_zero_coordinates = (2, 4, 5, 6)
        self.position_zero_width = len(self.position_zero_coordinates) - 1

        # Position zero retains the four verified coordinate anchors and fixes
        # its remaining scalar direction. Every later position is represented
        # in the full zero-mean feature subspace.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = self.position_zero_width + (
            num_embeddings - 1
        ) * (embedding_dim - 1)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift",
            base.weight.new_zeros(embedding_dim),
            persistent=False,
        )

        position_zero_basis = torch.zeros(
            len(self.position_zero_coordinates),
            self.position_zero_width,
        )
        for column in range(self.position_zero_width):
            scale = math.sqrt((column + 1) * (column + 2))
            position_zero_basis[: column + 1, column] = 1.0 / scale
            position_zero_basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer(
            "position_zero_basis", position_zero_basis, persistent=False
        )
=======
        self.position_zero_coordinates = (2, 4, 5, 6)
        self.position_zero_width = len(self.position_zero_coordinates) - 2

        # Position zero retains the four verified coordinate anchors, removes
        # the common direction, and removes the balanced 2-versus-4 contrast.
        # Every later position remains in the full zero-mean feature subspace.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = self.position_zero_width + (
            num_embeddings - 1
        ) * (embedding_dim - 1)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift",
            base.weight.new_zeros(embedding_dim),
            persistent=False,
        )

        position_zero_contrast = torch.zeros(
            len(self.position_zero_coordinates)
        )
        position_zero_contrast[0] = 1.0 / math.sqrt(2.0)
        position_zero_contrast[1] = -1.0 / math.sqrt(2.0)
        self.register_buffer(
            "position_zero_contrast",
            position_zero_contrast,
            persistent=False,
        )

        # Retain Helmert columns one and two; column zero is exactly the
        # normalized contrast between coordinates 2 and 4.
        position_zero_basis = torch.zeros(
            len(self.position_zero_coordinates),
            self.position_zero_width,
        )
        for column in range(self.position_zero_width):
            helmert_column = column + 1
            scale = math.sqrt(
                (helmert_column + 1) * (helmert_column + 2)
            )
            position_zero_basis[
                : helmert_column + 1, column
            ] = 1.0 / scale
            position_zero_basis[
                helmert_column + 1, column
            ] = -(helmert_column + 1) / scale
        self.register_buffer(
            "position_zero_basis", position_zero_basis, persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                remaining_shift = full[
                    0, list(module.position_zero_coordinates)
                ].mean().clone()
                for coordinate in module.position_zero_coordinates:
                    full[:, coordinate].sub_(remaining_shift)

                token_shifts = torch.stack(
                    (
                        anchor_shifts[0],
                        anchor_shifts[1],
                        remaining_shift,
                        anchor_shifts[2],
                        remaining_shift,
                        remaining_shift,
                        remaining_shift,
                        anchor_shifts[3],
                    )
                )

                full[1:].sub_(full[1:].mean(dim=1, keepdim=True))
                position_zero = (
                    full[0, list(module.position_zero_coordinates)]
                    @ module.position_zero_basis
                )
=======
                remaining_values = full[
                    0, list(module.position_zero_coordinates)
                ].clone()
                remaining_shift = remaining_values.mean()
                contrast_shift = (
                    remaining_values @ module.position_zero_contrast
                ) * module.position_zero_contrast
                remaining_shifts = remaining_shift + contrast_shift
                for coordinate, shift in zip(
                    module.position_zero_coordinates, remaining_shifts
                ):
                    full[:, coordinate].sub_(shift)

                token_shifts = torch.stack(
                    (
                        anchor_shifts[0],
                        anchor_shifts[1],
                        remaining_shifts[0],
                        anchor_shifts[2],
                        remaining_shifts[1],
                        remaining_shifts[2],
                        remaining_shifts[3],
                        anchor_shifts[3],
                    )
                )

                full[1:].sub_(full[1:].mean(dim=1, keepdim=True))
                position_zero = (
                    full[0, list(module.position_zero_coordinates)]
                    @ module.position_zero_basis
                )
>>>>>>> REPLACE