MECHANISM: Rank-seven learned lexical subspace

HYPOTHESIS: The verified 1,567-parameter quotient-residual, `d_ff=11` transformer will retain at least 99% accuracy when its full-rank tied token matrix is replaced by a learned rank-seven factorization, reducing the model to 1,509 parameters; this challenges the prior assumption that token representation and decoding require all eight hidden dimensions.

INTENDED_EDIT: Apply the verified quotient residual projections and `d_ff=11` configuration, then replace the tied 114×8 embedding/head with learned 114×7 token codes and a learned 7×8 projection used for both input embedding and output logits.

EVIDENCE: Reference Design 3 achieved 99.95% accuracy with 1,567 parameters, establishing that the quotient-residual architecture and eleven-unit MLP are viable. Its final normalized state varies on a seven-dimensional manifold, making a learned rank-seven lexical subspace a plausible, clean test of whether the full-rank tied vocabulary map is the remaining overprovisioned component.

<<<<<<< SEARCH
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.coeff(positions) @ self.basis.transpose(0, 1)


class CausalSelfAttention(nn.Module):
=======
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.coeff(positions) @ self.basis.transpose(0, 1)


class QuotientOutputLinear(nn.Module):
    """Linear map into the zero-mean feature subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")

        self.coeff = nn.Linear(in_features, out_features - 1, bias=bias)

        basis = torch.zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(x) @ self.basis.transpose(0, 1)


class FactorizedTokenEmbedding(nn.Module):
    """Learned low-rank token embedding shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if not 0 < rank < embedding_dim:
            raise ValueError("rank must be between zero and embedding_dim")

        self.code = nn.Embedding(num_embeddings, rank)
        self.proj = nn.Linear(rank, embedding_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.code(tokens))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.proj.weight.transpose(0, 1))
        return F.linear(latent, self.code.weight)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = QuotientOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = QuotientPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
        self.pos_emb = QuotientPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)
        nn.init.orthogonal_(self.token_emb.proj.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE