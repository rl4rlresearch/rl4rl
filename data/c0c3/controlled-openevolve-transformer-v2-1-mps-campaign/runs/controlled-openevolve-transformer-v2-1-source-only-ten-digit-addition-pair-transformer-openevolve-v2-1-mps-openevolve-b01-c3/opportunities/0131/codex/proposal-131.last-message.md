MECHANISM: Rank-five tied vocabulary bottleneck

HYPOTHESIS: A 749-parameter transformer will maintain at least 99% accuracy because the verified rank-six, 862-parameter design achieved 99.99%; reducing only its learned vocabulary feature rank by one tests whether that high-margin representation retains another 113-parameter compression.

INTENDED_EDIT: Adopt the verified final-pair gain sharing and reduce the centered tied vocabulary embedding from six to five learned feature coordinates, preserving the six-unit MLP, eight-dimensional residual stream, causal attention, training procedure, and protected decoding interface.

EVIDENCE: The rank-six vocabulary bottleneck reached 99.99% at 862 parameters, while the five-unit MLP failed at 96.48%; this motivates compressing the successful vocabulary codebook rather than the demonstrated load-bearing nonlinear width.

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
    """Rank-five tied embeddings centered across vocabulary and features."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, embedding_dim - 3)
        self.register_buffer(
            "vocab_basis", mean_zero_basis(num_embeddings), persistent=False
        )
        self.register_buffer(
            "feature_basis",
            mean_zero_basis(embedding_dim)[:, :-2],
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                self.secondary_value_gain,
            ),
            dim=0,
        )
=======
        y = att @ v
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain,
                self.secondary_value_gain[..., -1:],
            ),
            dim=-1,
        )
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                secondary_value_gain,
            ),
            dim=0,
        )
>>>>>>> REPLACE