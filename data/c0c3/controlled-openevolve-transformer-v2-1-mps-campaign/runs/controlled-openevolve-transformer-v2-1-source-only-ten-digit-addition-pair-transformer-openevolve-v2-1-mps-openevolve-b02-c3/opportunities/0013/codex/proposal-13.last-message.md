MECHANISM: Symmetric vocabulary-translation gauge fixing

HYPOTHESIS: Representing the rank-seven token codes in a fixed zero-sum vocabulary basis will reduce the qualified 1,460-parameter model to 1,453 parameters while retaining at least 99% accuracy, because it removes the same seven-dimensional translation redundancy as anchoring one token but keeps every vocabulary entry symmetrically trainable.

INTENDED_EDIT: Apply the qualified gauge-fixed rank-seven lexical factorization, then parameterize its token-code matrix with 113 orthonormal zero-sum vocabulary coefficients instead of 114 unconstrained rows.

EVIDENCE: The gauge-fixed rank-seven design achieved 99.96% accuracy with 1,460 parameters. Anchoring token zero reduced accuracy to 96.83%, motivating a symmetric gauge choice that removes the redundant shared translation without freezing or privileging any token.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class FactorizedTokenEmbedding(nn.Module):
    """Gauge-fixed low-rank token map with zero-mean vocabulary codes."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank != embedding_dim - 1:
            raise ValueError("rank must equal embedding_dim - 1")
        if num_embeddings < 2:
            raise ValueError("num_embeddings must be at least two")

        self.code_coeff = nn.Embedding(num_embeddings - 1, rank)

        token_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for col in range(num_embeddings - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            token_basis[: col + 1, col] = 1.0 / scale
            token_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("token_basis", token_basis, persistent=False)

        feature_basis = torch.zeros(embedding_dim, rank)
        for col in range(rank):
            scale = math.sqrt((col + 1) * (col + 2))
            feature_basis[: col + 1, col] = 1.0 / scale
            feature_basis[col + 1, col] = -(col + 1) / scale
        normal = torch.ones(embedding_dim) / math.sqrt(embedding_dim)
        self.register_buffer("feature_basis", feature_basis, persistent=False)
        self.register_buffer("normal", normal, persistent=False)

        self.tilt = nn.Parameter(torch.zeros(rank))

    def code_weight(self) -> torch.Tensor:
        return self.token_basis @ self.code_coeff.weight

    def projection_weight(self) -> torch.Tensor:
        return self.feature_basis + torch.outer(self.normal, self.tilt)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        codes = F.embedding(tokens, self.code_weight())
        return F.linear(codes, self.projection_weight())

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.projection_weight().transpose(0, 1))
        return F.linear(latent, self.code_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
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
        self.cfg = cfg
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
        self.pos_emb = QuotientPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE