MECHANISM: Six-dimensional learned positional routing subspace

HYPOTHESIS: Constraining positional embeddings from seven to six learned mean-zero coordinates will produce a 1,390-parameter model with at least 99% accuracy, because the verified 1,413-parameter design has 99.96% accuracy and this preserves substantially more routing capacity than the unsuccessful four-dimensional design.

INTENDED_EDIT: Apply the verified two-sided centered token embeddings and fixed 0.02 query bias, then remove one learned coordinate from every positional embedding.

EVIDENCE: The fixed-bias, two-sided embedding design achieved 99.96% with 1,413 parameters, while four-dimensional positional codes achieved only 92.27%; a one-coordinate positional ablation is the conservative next test between those results.

<<<<<<< SEARCH
class MeanZeroEmbedding(nn.Embedding):
    """Embedding parameterized on the subspace orthogonal to all-ones."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim - 1)
        self.register_buffer("basis", mean_zero_basis(embedding_dim), persistent=False)
=======
class MeanZeroEmbedding(nn.Embedding):
    """Embedding parameterized in a six-dimensional mean-zero subspace."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim - 2)
        self.register_buffer(
            "basis", mean_zero_basis(embedding_dim)[:, :-1], persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class VocabCenteredEmbedding(nn.Embedding):
    """Tied embeddings with zero mean across vocabulary entries."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, embedding_dim)
        self.register_buffer("basis", mean_zero_basis(num_embeddings), persistent=False)

    def full_weight(self) -> torch.Tensor:
        return self.basis @ self.weight
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(1))
        self.proj = MeanZeroLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = MeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE