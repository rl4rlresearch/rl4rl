MECHANISM: Transpose-tied learned ridge MLP

HYPOTHESIS: The resulting 816-parameter transformer will maintain at least 99% accuracy because the qualified 852-parameter model achieved 99.97%, while transpose tying preserves all six learned GELU features and the MLP’s maximum rank, adding per-feature gains instead of removing the load-bearing hidden unit that prior width reduction lost.

INTENDED_EDIT: Adopt the qualified 852-parameter rank-six embedding, attention, normalization, and far-bias design, then replace the MLP’s independent synthesis matrix with the transpose of its learned analysis matrix plus six learned gains.

EVIDENCE: The 852-parameter design reached 99.97%. Earlier MLP compression failed by reducing width, motivating a structurally different compression that retains six nonlinear detectors but challenges the assumption that their read and write directions must be independently learned.

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
    """Layer normalization with one shared scale sextet and fixed endpoints."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight.new_ones(1),
                self.weight.expand(x.size(-1) - 2),
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 5))
        self.register_buffer(
            "output_basis", mean_zero_basis(d_model), persistent=False
        )
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
        self.far_rel_bias = nn.Parameter(torch.zeros(11))
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 2)
        )
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 5))
        self.register_buffer(
            "output_basis", mean_zero_basis(d_model), persistent=False
        )
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
        self.far_rel_bias = nn.Parameter(torch.zeros(9))
>>>>>>> REPLACE

<<<<<<< SEARCH
        learned_bias = torch.cat(
            (
                self.rel_bias,
                self.far_rel_bias.expand(self.n_head, 11),
            ),
            dim=1,
        )
=======
        far_rel_bias = torch.cat(
            (self.far_rel_bias, self.far_rel_bias[-1:].expand(2))
        )
        learned_bias = torch.cat(
            (
                self.rel_bias,
                far_rel_bias.expand(self.n_head, 11),
            ),
            dim=1,
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
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = ScalarBiasMeanZeroLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class MLP(nn.Module):
    """Six learned ridge features with transpose-tied synthesis directions."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.output_gain = nn.Parameter(torch.ones(d_ff))
        self.output_bias = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "output_basis", mean_zero_basis(d_model), persistent=False
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x)) * self.output_gain
        bias = self.output_bias.expand(self.output_basis.size(1))
        reduced = F.linear(
            hidden,
            self.fc1.linear.weight.transpose(0, 1),
            bias,
        )
        return self.drop(reduced @ self.output_basis.transpose(0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=8)
=======
    p.add_argument("--d-ff", type=int, default=6)
>>>>>>> REPLACE