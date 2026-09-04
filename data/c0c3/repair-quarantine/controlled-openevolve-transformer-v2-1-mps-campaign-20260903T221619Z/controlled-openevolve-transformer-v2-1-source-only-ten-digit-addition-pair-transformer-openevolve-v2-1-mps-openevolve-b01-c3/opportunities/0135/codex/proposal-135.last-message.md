MECHANISM: Rank-six vocabulary bottleneck with two-pair final normalization-scale sharing

HYPOTHESIS: The resulting 859-parameter transformer will maintain at least 99% accuracy because the otherwise identical 860-parameter design achieved 99.88%, and this patch removes only one additional final LayerNorm scale through an adjacent-pair tie.

INTENDED_EDIT: Adopt the verified rank-six embedding and five-gain attention configuration, then expand five learned final LayerNorm scales as `[a,b,c,d,d,e,e,1]`.

EVIDENCE: The 860-parameter rank-six, five-gain model exceeded the requirement at 99.88%; further attention-gain compression failed sharply, so another isolated normalization-scale tie is the most informative lower-risk compression.

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
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with all but one scale learned."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with two tied scale pairs and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:3],
                self.weight[3:].repeat_interleave(2),
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 3)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 2)
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
                self.secondary_value_gain[..., :3],
                self.secondary_value_gain[..., 3:].repeat_interleave(2, dim=-1),
            ),
            dim=-1,
        )
>>>>>>> REPLACE