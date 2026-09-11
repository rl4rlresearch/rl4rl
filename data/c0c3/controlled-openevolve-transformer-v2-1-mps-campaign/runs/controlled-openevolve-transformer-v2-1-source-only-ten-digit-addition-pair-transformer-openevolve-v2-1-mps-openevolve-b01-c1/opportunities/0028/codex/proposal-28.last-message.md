MECHANISM: Incremental joint token–position embedding gauge quotient

HYPOTHESIS: Removing one additional common-token mean-free feature direction and transferring it to every positional embedding will reduce the model from 1,585 to 1,584 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Extend the tied embedding quotient from one to two dimensions, omitting one common-token feature contrast and transferring its initialized value into the positional embeddings without changing the model’s initial function.

EVIDENCE: The one-dimensional tied-embedding quotient achieved 99.97% at 1,585 parameters, while removing all seven remaining joint-gauge directions at once failed at 24.71%; removing exactly one is the smallest test of whether that failure was caused by the wholesale reparameterization.

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
    """Tied embedding modulo two exact token/output gauge directions."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()

        vocab_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            vocab_basis[: j + 1, j] = 1.0 / scale
            vocab_basis[j + 1, j] = -(j + 1) / scale

        feature_basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for j in range(embedding_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            feature_basis[: j + 1, j] = 1.0 / scale
            feature_basis[j + 1, j] = -(j + 1) / scale

        varying = torch.kron(vocab_basis, torch.eye(embedding_dim))
        common = torch.kron(
            torch.full((num_embeddings, 1), 1.0 / math.sqrt(num_embeddings)),
            feature_basis[:, 1:],
        )
        basis = torch.cat((varying, common), dim=1)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer(
            "transfer_direction", feature_basis[:, 0], persistent=False
        )
        self.register_buffer(
            "initial_transfer", torch.zeros(embedding_dim), persistent=False
        )

        self.weight = nn.Parameter(
            (full_weight.reshape(-1) @ basis).clone()
        )

    def full_weight(self) -> torch.Tensor:
        return (self.basis @ self.weight).view(
            self.num_embeddings, self.embedding_dim
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
    def __init__(
        self,
        in_features: int,
        out_features: int,
        embedding: MeanFreeTokenEmbedding,
    ):
        # Preserve the constructor draw made by the original output Linear.
        super().__init__(in_features, out_features, bias=False)
        del self.weight
        object.__setattr__(self, "embedding", embedding)
=======
    def __init__(
        self,
        in_features: int,
        out_features: int,
        embedding: MeanFreeTokenEmbedding,
        position_embedding: MeanFreePositionEmbedding,
    ):
        # Preserve the constructor draw made by the original output Linear.
        super().__init__(in_features, out_features, bias=False)
        del self.weight
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "position_embedding", position_embedding)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.lm_head = TiedMeanFreeOutput(
            cfg.d_model, cfg.vocab_size, self.token_emb
        )
=======
        self.lm_head = TiedMeanFreeOutput(
            cfg.d_model, cfg.vocab_size, self.token_emb, self.pos_emb
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                nn.init.normal_(full, mean=0.0, std=0.02)
                embedding.weight.copy_(full.reshape(-1) @ embedding.basis)
                transfer = (
                    full.mean(dim=0) @ embedding.transfer_direction
                ) * embedding.transfer_direction
                embedding.initial_transfer.copy_(transfer)
                position = module.position_embedding
                position.weight.add_(transfer @ position.basis)
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
                transfer = (
                    full.mean(dim=0) @ module.transfer_direction
                ) * module.transfer_direction
                module.initial_transfer.copy_(transfer)
>>>>>>> REPLACE