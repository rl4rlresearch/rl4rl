MECHANISM: Rank-six tied vocabulary bottleneck

HYPOTHESIS: The resulting 864-parameter transformer will maintain at least 99% accuracy because the six-unit MLP reached 100% while the 973-parameter gain-tied model reached 99.93%; retaining that nonlinear width but constraining the previously untested token/logit codebook to six centered features should preserve the learned computation.

INTENDED_EDIT: Replace the full seven-coordinate centered vocabulary embedding with a learned six-coordinate embedding projected through a fixed orthonormal mean-zero basis, removing 111 parameters while preserving the eight-dimensional residual stream, six-unit MLP, causal attention, and tied learned logits.

EVIDENCE: The five-unit MLP failed at 96.48%, identifying nonlinear width as load-bearing, whereas six units repeatedly exceeded 99%. The old designs implicitly assumed all seven learned vocabulary-feature coordinates were necessary; this patch instead tests a lower-rank learned input/output representation without revisiting the failed MLP bottleneck.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE