MECHANISM: Rank-five vocabulary bottleneck with four-way secondary-head gain tying

HYPOTHESIS: Combining the verified rank-five embedding with the independently verified `[a,b,b,c,c,d,d]` gain pattern will achieve at least 99% accuracy with 747 parameters.

INTENDED_EDIT: Reduce the tied embedding from rank six to rank five and tie one additional secondary-head gain while retaining the six-unit MLP.

EVIDENCE: The 748-parameter rank-five design achieved 99.88% accuracy, whereas reducing its MLP width failed; Reference Design 2’s four-parameter secondary-gain pattern achieved 99.93%, motivating this orthogonal one-parameter reduction.

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
            torch.ones(n_head - 1, self.output_dim - 2)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :3],
                self.secondary_value_gain[..., 3:].repeat_interleave(2, dim=-1),
            ),
            dim=-1,
        )
=======
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :1],
                self.secondary_value_gain[..., 1:].repeat_interleave(2, dim=-1),
            ),
            dim=-1,
        )
>>>>>>> REPLACE