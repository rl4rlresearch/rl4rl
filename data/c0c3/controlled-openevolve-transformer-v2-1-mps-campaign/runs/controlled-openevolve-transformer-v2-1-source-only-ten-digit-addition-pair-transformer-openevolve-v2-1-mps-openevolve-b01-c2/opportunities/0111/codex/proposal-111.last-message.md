MECHANISM: Tied-token Helmert gauge for the final token–position translation

HYPOTHESIS: Anchoring the first token embedding’s projection onto `(2 + 4 - 2*5)` while retaining that learned positional contrast will reduce the qualified model from 1,509 to 1,508 parameters and preserve at least 99% accuracy.

INTENDED_EDIT: Remove one exact token–position translation degree of freedom from the tied token embedding, transfer its initialization component into every positional row, and use an orthonormal chart for the affected token coordinates.

EVIDENCE: The current 1,509-parameter design achieved 99.98%, whereas deleting its retained positional contrast previously reached only 96.98%; moving the same exact quotient to the tied-token side preserves the sensitive learned positional coordinate, and prior Helmert charts successfully recovered optimization quality for query row 6.

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
        self.contrast_coordinates = (2, 4, 5, 6)

        # Fix the existing scalar gauge and the first token's projection onto
        # the positional (2 + 4 - 2*5) contrast.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 2
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_position_shift", base.weight.new_zeros(()), persistent=False
        )

        contrast_direction = torch.zeros(len(self.contrast_coordinates))
        contrast_direction[:2] = 1.0 / math.sqrt(6.0)
        contrast_direction[2] = -2.0 / math.sqrt(6.0)
        self.register_buffer(
            "contrast_direction", contrast_direction, persistent=False
        )

        # These orthonormal columns span the complement of the removed
        # contrast within coordinates 2, 4, 5, and 6.
        contrast_basis = torch.zeros(len(self.contrast_coordinates), 3)
        contrast_basis[0, 0] = 1.0 / math.sqrt(2.0)
        contrast_basis[1, 0] = -1.0 / math.sqrt(2.0)
        contrast_basis[:3, 1] = 1.0 / math.sqrt(12.0)
        contrast_basis[3, 1] = -3.0 / math.sqrt(12.0)
        contrast_basis[:, 2] = 0.5
        self.register_buffer(
            "contrast_basis", contrast_basis, persistent=False
        )

    @torch.no_grad()
    def _copy_dense_weight(self, full: torch.Tensor) -> None:
        direct_zero = full[0, [1, 3, 7]]
        reduced_zero = (
            full[0, list(self.contrast_coordinates)] @ self.contrast_basis
        )
        self.weight.copy_(
            torch.cat((direct_zero, reduced_zero, full[1:].flatten()))
        )

    def dense_weight(self) -> torch.Tensor:
        direct_zero = self.weight[:3]
        reduced_zero = self.weight[3:6]
        contrast_zero = self.contrast_basis @ reduced_zero
        zero = self.weight.new_zeros(())
        token_zero = torch.stack(
            (
                zero,
                direct_zero[0],
                contrast_zero[0],
                direct_zero[1],
                contrast_zero[1],
                contrast_zero[2],
                contrast_zero[3],
                direct_zero[2],
            )
        )
        remaining = self.weight[6:].view(
            self.num_embeddings - 1, self.embedding_dim
        )
        return torch.cat((token_zero.unsqueeze(0), remaining), dim=0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.dense_weight())

    @torch.no_grad()
    def transfer_coordinate_shifts(self, coordinates, shifts) -> None:
        full = self.dense_weight().clone()
        for coordinate, shift in zip(coordinates, shifts):
            full[:, coordinate].add_(shift)

        anchor = full[0, 0].clone()
        full.sub_(anchor)
        self._copy_dense_weight(full)
>>>>>>> REPLACE

<<<<<<< SEARCH
        dense = torch.cat(
            (position_zero.unsqueeze(0), remaining_positions), dim=0
        )
        return F.embedding(indices, dense)


class TinyDecoderLM(nn.Module):
=======
        dense = torch.cat(
            (position_zero.unsqueeze(0), remaining_positions), dim=0
        )
        return F.embedding(indices, dense)

    @torch.no_grad()
    def transfer_contrast_shift(self, shift: torch.Tensor) -> None:
        # Compensate the tied-token contrast gauge without removing the
        # sensitive learned position-zero contrast.
        self.weight[0].add_(shift)
        dense_shift = self.weight.new_zeros(self.embedding_dim)
        dense_shift[list(self.position_zero_coordinates)] = (
            self.position_zero_basis[:, 0] * shift
        )
        reduced_shift = dense_shift @ self.feature_basis
        self.weight[self.position_zero_width :].view(
            self.num_embeddings - 1, self.embedding_dim - 1
        ).add_(reduced_shift)


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        self.token_emb.transfer_coordinate_shifts(
            tuple(range(cfg.d_model)), self.pos_emb._init_token_shift
        )
=======
        self.apply(self._init_weights)

        self.pos_emb.transfer_contrast_shift(
            self.token_emb._init_position_shift
        )
        self.token_emb.transfer_coordinate_shifts(
            tuple(range(cfg.d_model)), self.pos_emb._init_token_shift
        )
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
                anchor = full[0, 0].clone()
                full.sub_(anchor)
                contrast_shift = (
                    full[0, list(module.contrast_coordinates)]
                    @ module.contrast_direction
                )
                for coordinate, direction in zip(
                    module.contrast_coordinates, module.contrast_direction
                ):
                    full[:, coordinate].sub_(contrast_shift * direction)
                module._copy_dense_weight(full)
                module._init_position_shift.copy_(contrast_shift)
        elif isinstance(module, TiedGaugedLMHead):
            with torch.no_grad():
                embedding = module._embedding
                full = embedding.weight.new_empty(
                    embedding.num_embeddings, embedding.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                anchor = full[0, 0].clone()
                full.sub_(anchor)
                contrast_shift = (
                    full[0, list(embedding.contrast_coordinates)]
                    @ embedding.contrast_direction
                )
                for coordinate, direction in zip(
                    embedding.contrast_coordinates,
                    embedding.contrast_direction,
                ):
                    full[:, coordinate].sub_(contrast_shift * direction)
                embedding._copy_dense_weight(full)
                embedding._init_position_shift.copy_(contrast_shift)
>>>>>>> REPLACE