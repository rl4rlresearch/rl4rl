MECHANISM: Rank-five vocabulary bottleneck with three-way dyadic secondary-head gain tying

HYPOTHESIS: Extending the successful secondary-head gain compression from `[a,b,b,c,c,d,d]` to `[a,b,b,c,c,c,c]` will reduce the verified 747-parameter rank-five model to 746 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Adopt the verified rank-five tied embedding and replace the secondary head’s four learned value gains with three gains expanded in groups of one, two, and four coordinates.

EVIDENCE: Reference Design 2 achieved 99.89% accuracy with 747 parameters, and its one-parameter gain reduction slightly exceeded the 748-parameter Reference Design 1; this motivates testing one further isolated gain tie while preserving the six-unit MLP whose reduction failed.

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
            torch.ones(n_head - 1, self.output_dim - 3)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :1],
                self.secondary_value_gain[..., 1:].repeat_interleave(2, dim=-1),
            ),
            dim=-1,
        )
=======
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :1],
                self.secondary_value_gain[..., 1:2].repeat_interleave(2, dim=-1),
                self.secondary_value_gain[..., 2:].repeat_interleave(4, dim=-1),
            ),
            dim=-1,
        )
>>>>>>> REPLACE