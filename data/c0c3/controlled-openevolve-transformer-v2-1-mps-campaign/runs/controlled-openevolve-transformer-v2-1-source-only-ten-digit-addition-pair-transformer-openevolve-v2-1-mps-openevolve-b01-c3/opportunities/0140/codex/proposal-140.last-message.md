MECHANISM: Nonlinear rank-restoring vocabulary manifold

HYPOTHESIS: The resulting 776-parameter transformer will maintain at least 99% accuracy because the linear rank-five codebook narrowly reached 98.50%, while a learned GELU lift can generate a full-rank six-feature embedding matrix from five per-token coordinates instead of permanently deleting the sixth feature direction.

INTENDED_EDIT: Replace the anchored dense rank-six vocabulary matrix with five-dimensional learned token codes decoded through a shared learned 5×6 GELU map, retaining tied learned input embeddings and output logits.

EVIDENCE: Rank six repeatedly exceeded 99% while fixed linear rank five missed by only 0.5 percentage points; this tests whether the load-bearing property is six-dimensional decoded features rather than six independent coefficients per vocabulary item.

<<<<<<< SEARCH
class VocabCenteredEmbedding(nn.Embedding):
    """Rank-six tied embeddings with one anchored latent coefficient."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, embedding_dim - 2)
        self.weight = nn.Parameter(self.weight.new_empty(self.weight.numel() - 1))
        self.register_buffer(
            "vocab_basis", mean_zero_basis(num_embeddings), persistent=False
        )
        self.register_buffer(
            "feature_basis",
            mean_zero_basis(embedding_dim)[:, :-1],
            persistent=False,
        )

    def full_weight(self) -> torch.Tensor:
        latent_weight = F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )
        return self.vocab_basis @ latent_weight @ self.feature_basis.transpose(0, 1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
=======
class VocabCenteredEmbedding(nn.Embedding):
    """Tied embeddings decoded from a learned nonlinear token manifold."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, embedding_dim - 3)
        self.feature_map = nn.Parameter(
            torch.empty(self.embedding_dim, self.embedding_dim + 1)
        )
        nn.init.orthogonal_(self.feature_map)
        self.register_buffer(
            "vocab_basis", mean_zero_basis(num_embeddings), persistent=False
        )
        self.register_buffer(
            "feature_basis",
            mean_zero_basis(embedding_dim)[:, :-1],
            persistent=False,
        )

    def full_weight(self) -> torch.Tensor:
        latent_weight = 2.0 * F.gelu(self.weight @ self.feature_map)
        return self.vocab_basis @ latent_weight @ self.feature_basis.transpose(0, 1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
>>>>>>> REPLACE