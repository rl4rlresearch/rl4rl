MECHANISM: Single-coefficient anchored rank-six vocabulary codebook

HYPOTHESIS: The resulting 860-parameter transformer will maintain at least 99% accuracy because it preserves the successful rank-six embedding while fixing only one of its 678 latent coefficients, rather than removing the entire 113-coefficient feature column that reduced accuracy to 98.50%.

INTENDED_EDIT: Store one fewer learned vocabulary-codebook coefficient and append a fixed zero when reconstructing the unchanged rank-six latent matrix.

EVIDENCE: The current 861-parameter rank-six model achieved 99.99% accuracy, whereas reducing the whole embedding to rank five removed 113 parameters and narrowly failed; this isolates the smallest possible embedding compression without reducing its rank.

<<<<<<< SEARCH
class VocabCenteredEmbedding(nn.Embedding):
    """Rank-six tied embeddings centered across vocabulary and features."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, embedding_dim - 2)
        self.register_buffer(
            "vocab_basis", mean_zero_basis(num_embeddings), persistent=False
        )
        self.register_buffer(
            "feature_basis",
            mean_zero_basis(embedding_dim)[:, :-1],
            persistent=False,
        )

    def full_weight(self) -> torch.Tensor:
        return self.vocab_basis @ self.weight @ self.feature_basis.transpose(0, 1)
=======
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
>>>>>>> REPLACE