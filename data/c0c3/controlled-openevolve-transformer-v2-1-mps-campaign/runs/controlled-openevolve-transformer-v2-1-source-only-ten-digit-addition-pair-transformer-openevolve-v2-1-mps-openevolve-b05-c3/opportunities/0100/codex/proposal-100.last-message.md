MECHANISM: Learned rank-seven tied token manifold

HYPOTHESIS: Replacing the assumed full-rank eight-coordinate token table with learned seven-dimensional token codes and a shared learned basis will reduce the model from 1,256 to 1,200 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Factorize the tied input/output embedding into a learned rank-seven codebook and learned 7×8 basis, initialize it as a stable seven-channel embedding, and train both factors with regular AdamW.

EVIDENCE: The 1,256-parameter fixed-spacing design achieved 99.94% accuracy, while every available design retains a full-rank tied embedding and concentrates reductions in routing or normalization; this tests that unchallenged representation assumption with substantially more potential progress.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Embedding):
    """Tied embedding represented modulo one global scalar shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat((self.weight, self.weight.new_zeros(1)))
        return flat.view(self.num_embeddings, self.embedding_dim)
=======
class GaugeFixedEmbedding(nn.Embedding):
    """Tied embedding represented by learned codes in a learned low-rank basis."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.rank = embedding_dim - 1
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, self.rank)
        )
        self.basis = nn.Parameter(
            torch.empty(self.rank, embedding_dim)
        )

    def full_weight(self) -> torch.Tensor:
        return self.weight @ self.basis
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                full.sub_(full[-1, -1].clone())
                module.weight.copy_(full.reshape(-1)[:-1])
=======
        if isinstance(module, GaugeFixedEmbedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.basis.zero_()
                module.basis[:, : module.rank].copy_(
                    torch.eye(
                        module.rank,
                        device=module.basis.device,
                        dtype=module.basis.dtype,
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeTiedHead):
            embedding = module._embedding
            full = embedding.weight.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                full.sub_(full[-1, -1].clone())
                embedding.weight.copy_(full.reshape(-1)[:-1])
=======
        elif isinstance(module, GaugeTiedHead):
            embedding = module._embedding
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.basis.zero_()
                embedding.basis[:, : embedding.rank].copy_(
                    torch.eye(
                        embedding.rank,
                        device=embedding.basis.device,
                        dtype=embedding.basis.dtype,
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
    embedding_gauge_parameters = [
        model.token_emb.weight
    ]
=======
    embedding_gauge_parameters = []
>>>>>>> REPLACE