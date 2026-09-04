MECHANISM: Rank-six vocabulary bottleneck with three-pair secondary-head gain sharing

HYPOTHESIS: The resulting 860-parameter transformer will maintain at least 99% accuracy because the rank-six 862-parameter design achieved 99.99%, while the `[a,b,b,c,c,d,d]` gain pattern independently achieved 99.93% with the same six-unit MLP.

INTENDED_EDIT: Adopt the verified rank-six tied vocabulary embedding and six-unit MLP, then represent seven secondary-head gains with four learned values expanded as `[a,b,b,c,c,d,d]`.

EVIDENCE: Rank five failed at 98.50%, so this preserves the successful rank-six embedding; the rank-six final-pair-tied model reached 99.99%, and three-pair gain sharing already exceeded the accuracy threshold in the 973-parameter design.

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

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 3)
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
                self.secondary_value_gain[..., :1],
                self.secondary_value_gain[..., 1:].repeat_interleave(2, dim=-1),
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

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=8)
=======
    p.add_argument("--d-ff", type=int, default=6)
>>>>>>> REPLACE