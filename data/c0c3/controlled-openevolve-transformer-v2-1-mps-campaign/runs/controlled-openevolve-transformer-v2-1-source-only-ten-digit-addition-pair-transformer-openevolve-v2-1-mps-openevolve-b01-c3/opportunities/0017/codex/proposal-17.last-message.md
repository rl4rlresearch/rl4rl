MECHANISM: Cross-head query-bias scale gauge tying

HYPOTHESIS: Tying the two scalar query biases will reduce the verified 1,528-parameter design to 1,527 parameters while retaining at least 99% accuracy, because each head’s unrestricted query and key projections permit reciprocal rescaling that normalizes nonzero bias amplitudes to one shared learned value without changing attention logits.

INTENDED_EDIT: Apply the verified vocabulary-centered embeddings, mean-zero residual projections, anchored terminal normalization, and scalar query-bias gauge reductions, then share one learned query-bias scalar across both attention heads.

EVIDENCE: The independent-scalar query-bias design achieved 99.86% accuracy with 1,528 parameters; sharing only their amplitudes preserves full query/key widths and tests a remaining headwise scaling redundancy without repeating the failed capacity reductions.

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


class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with all but one scale learned."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(1))
        self.proj = MeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(
            self.q_bias.expand(self.n_head, 1), (0, self.head_dim - 1)
        ).reshape(d_model)
        q = q + q_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = MeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.token_emb = VocabCenteredEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = AnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(x, self.token_emb.full_weight())
>>>>>>> REPLACE