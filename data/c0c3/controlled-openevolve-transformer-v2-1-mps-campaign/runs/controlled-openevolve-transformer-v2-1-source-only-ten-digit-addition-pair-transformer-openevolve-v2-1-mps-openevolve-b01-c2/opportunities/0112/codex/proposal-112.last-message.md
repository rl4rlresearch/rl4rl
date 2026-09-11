MECHANISM: Lightweight tied-token Helmert translation quotient

HYPOTHESIS: Anchoring token zero’s projection onto `(2 + 4 - 2*5)` while retaining that positional contrast will produce a 1,508-parameter model with at least 99% accuracy and complete within the verification limit.

INTENDED_EDIT: Reproduce the qualified 1,509-parameter single-contrast position chart, then remove the remaining exact token–position translation using a small orthonormal token-row chart, transferring the omitted initialization component into every positional row and retaining ordinary AdamW.

EVIDENCE: The single-contrast 1,509-parameter design achieved 99.98%. The prior 1,508 tied-token experiment timed out rather than failing accuracy, so this lower-overhead formulation tests the same exact quotient without a custom token optimizer.

<<<<<<< SEARCH
class ScalarGaugedTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 1
        self.weight = nn.Parameter(base.weight.new_empty(retained))

    def dense_weight(self) -> torch.Tensor:
        flat = torch.cat((self.weight.new_zeros(1), self.weight))
        return flat.view(self.num_embeddings, self.embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.dense_weight())

    @torch.no_grad()
    def transfer_coordinate_shifts(self, coordinates, shifts) -> None:
        full = self.dense_weight().clone()
        for coordinate, shift in zip(coordinates, shifts):
            full[:, coordinate].add_(shift)

        anchor = full[0, 0].clone()
        full.sub_(anchor)
        self.weight.copy_(full.flatten()[1:])
=======
class ScalarGaugedTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.translation_coordinates = (2, 4, 5)
        self.direct_coordinates = (1, 3, 6, 7)
        self.first_token_width = embedding_dim - 2

        # Fix the existing scalar gauge and token zero's projection onto the
        # retained positional contrast. All remaining token rows stay dense.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 2
        self.weight = nn.Parameter(base.weight.new_empty(retained))

        translation_direction = torch.zeros(
            len(self.translation_coordinates)
        )
        scale = math.sqrt(6.0)
        translation_direction[:2] = 1.0 / scale
        translation_direction[2] = -2.0 / scale
        self.register_buffer(
            "translation_direction",
            translation_direction,
            persistent=False,
        )

        # These orthonormal columns span the complement of (1, 1, -2).
        token_zero_basis = torch.zeros(
            len(self.translation_coordinates),
            len(self.translation_coordinates) - 1,
        )
        token_zero_basis[0, 0] = 1.0 / math.sqrt(2.0)
        token_zero_basis[1, 0] = -1.0 / math.sqrt(2.0)
        token_zero_basis[:, 1] = 1.0 / math.sqrt(3.0)
        self.register_buffer(
            "token_zero_basis", token_zero_basis, persistent=False
        )
        self.register_buffer(
            "_init_position_shift",
            base.weight.new_zeros(embedding_dim),
            persistent=False,
        )

    def dense_weight(self) -> torch.Tensor:
        direct_width = len(self.direct_coordinates)
        direct = self.weight[:direct_width]
        reduced = self.weight[
            direct_width : self.first_token_width
        ]
        translated_coordinates = self.token_zero_basis @ reduced
        zero = self.weight.new_zeros(())
        first_token = torch.stack(
            (
                zero,
                direct[0],
                translated_coordinates[0],
                direct[1],
                translated_coordinates[1],
                translated_coordinates[2],
                direct[2],
                direct[3],
            )
        )
        remaining = self.weight[self.first_token_width :].view(
            self.num_embeddings - 1, self.embedding_dim
        )
        return torch.cat((first_token.unsqueeze(0), remaining), dim=0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.dense_weight())

    def _copy_dense_chart(self, full: torch.Tensor) -> None:
        direct = full[0, list(self.direct_coordinates)]
        reduced = (
            full[0, list(self.translation_coordinates)]
            @ self.token_zero_basis
        )
        self.weight.copy_(
            torch.cat((direct, reduced, full[1:].flatten()))
        )

    @torch.no_grad()
    def initialize_dense(self, full: torch.Tensor) -> None:
        full = full.clone()
        anchor = full[0, 0].clone()
        full.sub_(anchor)

        values = full[0, list(self.translation_coordinates)].clone()
        shift = (
            values @ self.translation_direction
        ) * self.translation_direction
        for coordinate, component in zip(
            self.translation_coordinates, shift
        ):
            full[:, coordinate].sub_(component)

        self._init_position_shift.zero_()
        for coordinate, component in zip(
            self.translation_coordinates, shift
        ):
            self._init_position_shift[coordinate].copy_(component)
        self._copy_dense_chart(full)

    @torch.no_grad()
    def transfer_coordinate_shifts(self, coordinates, shifts) -> torch.Tensor:
        full = self.dense_weight().clone()
        for coordinate, shift in zip(coordinates, shifts):
            full[:, coordinate].add_(shift)

        anchor = full[0, 0].clone()
        full.sub_(anchor)

        values = full[0, list(self.translation_coordinates)].clone()
        contrast_shift = (
            values @ self.translation_direction
        ) * self.translation_direction
        for coordinate, component in zip(
            self.translation_coordinates, contrast_shift
        ):
            full[:, coordinate].sub_(component)
        self._copy_dense_chart(full)

        position_shift = self._init_position_shift.clone()
        for coordinate, component in zip(
            self.translation_coordinates, contrast_shift
        ):
            position_shift[coordinate].add_(component)
        return position_shift
>>>>>>> REPLACE

<<<<<<< SEARCH
class FourCoordinateFullyRowScalarGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
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
=======
class SingleContrastFullyRowScalarGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.position_zero_coordinates = (2, 4, 5, 6)
        self.position_zero_width = len(self.position_zero_coordinates) - 3

        # Retain only the qualified (2 + 4 - 2*5) position-zero contrast.
        # Every later position remains in the zero-mean feature subspace.
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
        scale = math.sqrt(6.0)
        position_zero_basis[:2, 0] = 1.0 / scale
        position_zero_basis[2, 0] = -2.0 / scale
        self.register_buffer(
            "position_zero_basis", position_zero_basis, persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return F.embedding(indices, dense)


class TinyDecoderLM(nn.Module):
=======
        return F.embedding(indices, dense)

    @torch.no_grad()
    def add_common_shift(self, shift: torch.Tensor) -> None:
        start = self.position_zero_width
        coordinate_shift = shift[list(self.position_zero_coordinates)]
        self.weight[:start].add_(
            self.position_zero_basis.transpose(0, 1)
            @ coordinate_shift
        )

        reduced_rows = self.weight[start:].view(
            self.num_embeddings - 1, self.embedding_dim - 1
        )
        dense_rows = (
            reduced_rows @ self.feature_basis.transpose(0, 1)
        )
        dense_rows.add_(shift)
        self.weight[start:].copy_(
            (dense_rows @ self.feature_basis).flatten()
        )


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = FourCoordinateFullyRowScalarGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
=======
        self.pos_emb = SingleContrastFullyRowScalarGaugedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb.transfer_coordinate_shifts(
            tuple(range(cfg.d_model)), self.pos_emb._init_token_shift
        )
=======
        position_shift = self.token_emb.transfer_coordinate_shifts(
            tuple(range(cfg.d_model)), self.pos_emb._init_token_shift
        )
        self.pos_emb.add_common_shift(position_shift)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, ScalarGaugedTokenEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                anchor = full[0, 0].clone()
                full.sub_(anchor)
                module.weight.copy_(full.flatten()[1:])
        elif isinstance(module, TiedGaugedLMHead):
            with torch.no_grad():
                embedding = module._embedding
                full = embedding.weight.new_empty(
                    embedding.num_embeddings, embedding.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                anchor = full[0, 0].clone()
                full.sub_(anchor)
                embedding.weight.copy_(full.flatten()[1:])
=======
        if isinstance(module, ScalarGaugedTokenEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.initialize_dense(full)
        elif isinstance(module, TiedGaugedLMHead):
            with torch.no_grad():
                embedding = module._embedding
                full = embedding.weight.new_empty(
                    embedding.num_embeddings, embedding.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                embedding.initialize_dense(full)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(
            module, FourCoordinateFullyRowScalarGaugedPositionEmbedding
        ):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                anchor_coordinates = (0, 1, 3, 7)
                anchor_shifts = full[
                    0, list(anchor_coordinates)
                ].clone()
                for coordinate, shift in zip(
                    anchor_coordinates, anchor_shifts
                ):
                    full[:, coordinate].sub_(shift)

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
                reduced_rows = full[1:] @ module.feature_basis
                module.weight.copy_(
                    torch.cat((position_zero, reduced_rows.flatten()))
                )
                module._init_token_shift.copy_(token_shifts)
=======
        elif isinstance(
            module, SingleContrastFullyRowScalarGaugedPositionEmbedding
        ):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                anchor_coordinates = (0, 1, 3, 7)
                anchor_shifts = full[
                    0, list(anchor_coordinates)
                ].clone()
                for coordinate, shift in zip(
                    anchor_coordinates, anchor_shifts
                ):
                    full[:, coordinate].sub_(shift)

                remaining_values = full[
                    0, list(module.position_zero_coordinates)
                ].clone()
                retained_values = module.position_zero_basis @ (
                    module.position_zero_basis.transpose(0, 1)
                    @ remaining_values
                )
                remaining_shifts = remaining_values - retained_values
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
                reduced_rows = full[1:] @ module.feature_basis
                module.weight.copy_(
                    torch.cat((position_zero, reduced_rows.flatten()))
                )
                module._init_token_shift.copy_(token_shifts)
>>>>>>> REPLACE