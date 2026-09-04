MECHANISM: Joint token–position embedding gauge quotient

HYPOTHESIS: Centering the tied embedding across vocabulary independently in all eight feature coordinates will reduce the model from 1,585 to 1,578 parameters while retaining at least 99% accuracy, because the removed common token vector can be transferred to every positional embedding and changes output logits only by a softmax-null common offset.

INTENDED_EDIT: Replace the one-dimensional flattened embedding quotient with an eight-dimensional vocabulary-centering quotient, and transfer the removed mean-free initialization component into the positional embeddings.

EVIDENCE: The existing global tied-embedding quotient achieved 99.97% accuracy at 1,585 parameters, and mean-free positional embeddings have also retained high accuracy; together these support extending the proven quotient to the exact joint token–position gauge rather than pruning optimization-sensitive attention or LayerNorm capacity.

<<<<<<< SEARCH
class MeanFreeTokenEmbedding(nn.Embedding):
    """Tied embedding modulo its globally constant, functionally null shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()
        size = num_embeddings * embedding_dim

        basis = torch.zeros(size, size - 1)
        for j in range(size - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        self.weight = nn.Parameter(
            (full_weight.reshape(-1) @ basis).clone()
        )

    def full_weight(self) -> torch.Tensor:
        return (self.basis @ self.weight).view(
            self.num_embeddings, self.embedding_dim
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
=======
class MeanFreeTokenEmbedding(nn.Embedding):
    """Tied embedding modulo its common-across-token feature vector."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()

        basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        self.weight = nn.Parameter(
            (basis.transpose(0, 1) @ full_weight).clone()
        )
        self._removed_mean = full_weight.mean(dim=0).clone()

    def full_weight(self) -> torch.Tensor:
        return self.basis @ self.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, TiedMeanFreeOutput):
            # The original tied output Linear initialized the shared embedding a
            # second time. Preserve that draw and retain its quotient coordinates.
            embedding = module.embedding
            with torch.no_grad():
                full = torch.empty(
                    embedding.num_embeddings,
                    embedding.embedding_dim,
                    device=embedding.weight.device,
                    dtype=embedding.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                embedding.weight.copy_(full.reshape(-1) @ embedding.basis)
        elif isinstance(module, MeanFreeTokenEmbedding):
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(full.reshape(-1) @ module.basis)
=======
        if isinstance(module, TiedMeanFreeOutput):
            # The original tied output Linear initialized the shared embedding a
            # second time. Preserve that draw and retain its quotient coordinates.
            embedding = module.embedding
            with torch.no_grad():
                full = torch.empty(
                    embedding.num_embeddings,
                    embedding.embedding_dim,
                    device=embedding.weight.device,
                    dtype=embedding.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                embedding.weight.copy_(embedding.basis.transpose(0, 1) @ full)
                embedding._removed_mean = full.mean(dim=0)
        elif isinstance(module, MeanFreeTokenEmbedding):
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(module.basis.transpose(0, 1) @ full)
                module._removed_mean = full.mean(dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Centering removes the common token vector. Transfer its observable
        # mean-free component to every position so initialization is unchanged
        # up to per-position constant shifts removed by LayerNorm.
        with torch.no_grad():
            self.pos_emb.weight.add_(
                self.token_emb._removed_mean @ self.pos_emb.basis
            )
        del self.token_emb._removed_mean

    @staticmethod
>>>>>>> REPLACE