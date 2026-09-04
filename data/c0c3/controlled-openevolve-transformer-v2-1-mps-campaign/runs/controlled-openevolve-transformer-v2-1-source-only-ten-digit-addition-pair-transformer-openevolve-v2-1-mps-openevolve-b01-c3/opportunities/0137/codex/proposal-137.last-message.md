MECHANISM: Triple-shared final normalization scale

HYPOTHESIS: The resulting 858-parameter transformer will maintain at least 99% accuracy because the 859-parameter model achieved 99.65%, and extending its successful adjacent final-LayerNorm scale tie by one coordinate removes a single parameter without further constraining the sensitive rank-six vocabulary codebook.

INTENDED_EDIT: Adopt the verified single-coefficient-anchored rank-six embedding and five-gain attention pattern, then share three adjacent final LayerNorm scales instead of two, producing an 858-parameter model.

EVIDENCE: The 859-parameter pair-shared LayerNorm design achieved 99.65%, whereas anchoring a second embedding coefficient collapsed accuracy to 64.24%; this motivates applying the next isolated one-parameter reduction to normalization rather than the embedding.

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
    """Layer normalization with one shared scale triple and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight, self.weight[-1:].expand(2), self.weight.new_ones(1))
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