MECHANISM: Two-coordinate terminal-scale anchoring with verified gauge reductions

HYPOTHESIS: Applying the verified 1,534-parameter gauge-fixed design and fixing one additional final LayerNorm scale coordinate at 1 will produce a 1,533-parameter model with at least 99% accuracy, because the 1,534-parameter design achieved 100% accuracy and this removes only one relative terminal-scale degree of freedom while leaving six independently learned scales.

INTENDED_EDIT: Add vocabulary-centered tied embeddings and mean-zero residual projections, then use a bias-free final LayerNorm with six learned scale coordinates and two fixed anchors.

EVIDENCE: The seven-scale, one-anchor design achieved 100% accuracy with 1,534 parameters. The failed mean-one constraint coupled all scale coordinates, whereas this patch makes the smallest independent follow-up ablation by fixing only one additional coordinate.

<<<<<<< SEARCH
class MeanZeroEmbedding(nn.Embedding):
    """Embedding parameterized on the subspace orthogonal to all-ones."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim - 1)
        centered = torch.eye(embedding_dim) - torch.full(
            (embedding_dim, embedding_dim), 1.0 / embedding_dim
        )
        basis = torch.linalg.qr(centered[:, :-1], mode="reduced").Q
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return super().forward(idx) @ self.basis.transpose(0, 1)
=======
def mean_zero_basis(dim: int) -> torch.Tensor:
    centered = torch.eye(dim) - torch.full((dim, dim), 1.0 / dim)
    return torch.linalg.qr(centered[:, :-1], mode="reduced").Q


class MeanZeroEmbedding(nn.Embedding):
    """Embedding parameterized on the subspace orthogonal to all-ones."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim - 1)
        self.register_buffer("basis", mean_zero_basis(embedding_dim), persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return super().forward(idx) @ self.basis.transpose(0, 1)


class VocabCenteredEmbedding(nn.Embedding):
    """Tied embeddings with zero mean across vocabulary entries."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, embedding_dim)
        self.register_buffer("basis", mean_zero_basis(num_embeddings), persistent=False)

    def full_weight(self) -> torch.Tensor:
        return self.basis @ self.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


class TwoAnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with six learned scales and two fixed anchors."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(2)))
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = MeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = MeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.cfg = cfg
        self.token_emb = VocabCenteredEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = TwoAnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = F.linear(x, self.token_emb.full_weight())
>>>>>>> REPLACE