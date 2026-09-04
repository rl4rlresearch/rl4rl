MECHANISM: Septuple-shared final normalization scale

HYPOTHESIS: The resulting 854-parameter transformer will maintain at least 99% accuracy because the otherwise identical 855-parameter sextuple-shared design achieved 99.76%, and extending its shared final-LayerNorm scale group by one coordinate removes only one learned parameter.

INTENDED_EDIT: Adopt the qualified anchored rank-six vocabulary embedding, five-gain attention pattern, and six-unit MLP, then share all seven trainable final-LayerNorm scale coordinates while retaining the fixed scale anchor.

EVIDENCE: Successive triple-, quadruple-, quintuple-, and sextuple-shared LayerNorm designs achieved 99.93%, 99.68%, 99.94%, and 99.76%; another isolated normalization-scale tie is better supported than further compressing the embedding, attention gains, or MLP, which previously failed.

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
    """Bias-free LayerNorm with one fixed scale coordinate."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with one shared scale septet and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight, self.weight[-1:].expand(6), self.weight.new_ones(1))
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 2)
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
                self.secondary_value_gain[..., :3],
                self.secondary_value_gain[..., 3:].repeat_interleave(2, dim=-1),
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