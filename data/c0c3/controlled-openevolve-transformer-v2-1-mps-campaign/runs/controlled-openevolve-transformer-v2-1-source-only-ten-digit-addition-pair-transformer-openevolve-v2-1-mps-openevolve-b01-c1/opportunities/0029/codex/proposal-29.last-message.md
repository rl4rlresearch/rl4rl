MECHANISM: Position-centered joint token–position embedding gauge

HYPOTHESIS: Centering one shared positional-feature direction and transferring its initialization into the tied token embedding will produce 1,584 parameters while retaining at least 99% accuracy, because it removes an exact token–position gauge without constraining the token embedding’s learned common-feature capacity.

INTENDED_EDIT: Restore the successful full value bias, then remove one scalar from the positional-coordinate table with an orthonormal quotient and transfer the omitted common positional vector into every token embedding at initialization.

EVIDENCE: The full-value-bias, tied-embedding quotient reached 99.97% at 1,585 parameters, while pruning one value-bias coordinate reached only 97.31%. Removing additional common-token directions also failed, so this patch keeps those sensitive parameters and places the next exact gauge constraint on the already-successful mean-free positional representation.

<<<<<<< SEARCH
class MeanFreePositionEmbedding(nn.Module):
    """Learned positional vectors modulo LayerNorm-invariant constant offsets."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 1))

        basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for j in range(embedding_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        coordinates = F.embedding(idx, self.weight)
        return coordinates @ self.basis.transpose(0, 1)
=======
class MeanFreePositionEmbedding(nn.Module):
    """Mean-free positions with one common direction absorbed by token embeddings."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        feature_basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for j in range(embedding_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            feature_basis[: j + 1, j] = 1.0 / scale
            feature_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("feature_basis", feature_basis, persistent=False)

        coordinate_size = num_embeddings * (embedding_dim - 1)
        coordinate_basis = torch.zeros(coordinate_size, coordinate_size - 1)
        for j in range(coordinate_size - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            coordinate_basis[: j + 1, j] = 1.0 / scale
            coordinate_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("coordinate_basis", coordinate_basis, persistent=False)
        self.register_buffer(
            "initial_token_shift", torch.zeros(embedding_dim), persistent=False
        )
        self.weight = nn.Parameter(torch.empty(coordinate_size - 1))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        coordinates = (self.coordinate_basis @ self.weight).view(
            self.num_embeddings, self.embedding_dim - 1
        )
        coordinates = F.embedding(idx, coordinates)
        return coordinates @ self.feature_basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Keep constructor RNG consumption, then remove the softmax-null key bias
        # and prune one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model : -1])).clone()
        )
=======
        # Keep constructor RNG consumption and remove only the softmax-null key bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), F.pad(v_bias, (0, 1)))
        )
=======
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # The omitted positional direction is common to every position. Moving it
        # into every token vector preserves input sums exactly and changes output
        # logits only by a vocabulary-independent softmax-null scalar.
        with torch.no_grad():
            full_token_weight = (
                self.token_emb.full_weight()
                + self.pos_emb.initial_token_shift.unsqueeze(0)
            )
            self.token_emb.weight.copy_(
                full_token_weight.reshape(-1) @ self.token_emb.basis
            )

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then preserve its observable mean-free part.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(full @ module.basis)
=======
        elif isinstance(module, MeanFreePositionEmbedding):
            # Preserve the full-width draw, quotient one common positional
            # coordinate direction, and record the vector transferred to tokens.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                coordinates = full @ module.feature_basis
                module.weight.copy_(
                    coordinates.reshape(-1) @ module.coordinate_basis
                )
                module.initial_token_shift.copy_(
                    coordinates.mean() * module.feature_basis.sum(dim=1)
                )
>>>>>>> REPLACE