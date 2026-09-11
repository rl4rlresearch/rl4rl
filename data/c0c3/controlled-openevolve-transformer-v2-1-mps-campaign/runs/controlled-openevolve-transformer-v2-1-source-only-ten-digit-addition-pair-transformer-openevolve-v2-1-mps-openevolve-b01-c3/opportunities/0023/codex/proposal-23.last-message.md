MECHANISM: Two-sided orthonormal centering of tied token embeddings

HYPOTHESIS: Constraining tied token embeddings to the feature-mean-zero subspace will reduce the current 1,532-parameter model to 1,520 parameters while retaining at least 99% accuracy, because affine-free LayerNorm makes each token-dependent all-ones feature component invisible throughout the residual stack and terminal normalization, while terminal hidden states are mean-zero and therefore cannot use that component in output logits.

INTENDED_EDIT: Parameterize the tied embedding matrix with orthonormal bases across both vocabulary and feature dimensions, removing one functionally null feature coordinate for each of the twelve vocabulary-basis rows while preserving the current successful six-parameter query-bias design.

EVIDENCE: The 1,532-parameter current design achieved 99.94% accuracy with mean-zero positional embeddings and residual projections plus affine-free internal normalization. Unlike the failed asymmetric QKV and MLP reductions, this uses orthonormal coordinates on both sides and gives the same initialization distribution as projecting the existing embedding matrix onto its usable feature subspace.

<<<<<<< SEARCH
class VocabCenteredEmbedding(nn.Embedding):
    """Tied embeddings with zero mean across vocabulary entries."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, embedding_dim)
        self.register_buffer("basis", mean_zero_basis(num_embeddings), persistent=False)

    def full_weight(self) -> torch.Tensor:
        return self.basis @ self.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
=======
class VocabCenteredEmbedding(nn.Embedding):
    """Tied embeddings centered across vocabulary and feature dimensions."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, embedding_dim - 1)
        self.register_buffer(
            "vocab_basis", mean_zero_basis(num_embeddings), persistent=False
        )
        self.register_buffer(
            "feature_basis", mean_zero_basis(embedding_dim), persistent=False
        )

    def full_weight(self) -> torch.Tensor:
        return self.vocab_basis @ self.weight @ self.feature_basis.transpose(0, 1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())
>>>>>>> REPLACE