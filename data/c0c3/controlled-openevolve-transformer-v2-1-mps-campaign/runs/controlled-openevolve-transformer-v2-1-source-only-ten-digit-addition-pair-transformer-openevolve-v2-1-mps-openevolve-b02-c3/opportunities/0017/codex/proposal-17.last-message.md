MECHANISM: Second-direction pre-attention LayerNorm bias gauge fixing

HYPOTHESIS: Restricting the qualified 1,458-parameter model’s pre-attention LayerNorm bias from seven basis coefficients to six will yield 1,457 parameters while retaining at least 99% accuracy, because the omitted constant input shift is absorbable through query bias, key-softmax invariance, and attention-output bias.

INTENDED_EDIT: Apply the verified gauge-fixed rank-seven lexical, quotient-position, quotient-residual, `d_ff=11` architecture, while retaining only six learned pre-attention LayerNorm bias directions.

EVIDENCE: The 1,458-parameter design achieved 99.94% after removing one pre-attention LayerNorm bias direction. Removing one additional direction is the narrowest continuation of that successful mechanism, while preserving substantially more bias capacity than the failed complete LayerNorm-bias removal.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
=======
class QuotientPositionEmbedding(nn.Module):
    """Position embeddings modulo feature-wise all-ones shifts."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least two")

        self.coeff = nn.Embedding(num_embeddings, embedding_dim - 1)

        basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for col in range(embedding_dim - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

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
    """Gauge-fixed low-rank token map shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank != embedding_dim - 1:
            raise ValueError("rank must equal embedding_dim - 1")

        self.code = nn.Embedding(num_embeddings, rank)

        basis = torch.zeros(embedding_dim, rank)
        for col in range(rank):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        normal = torch.ones(embedding_dim) / math.sqrt(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("normal", normal, persistent=False)
        self.tilt = nn.Parameter(torch.zeros(rank))

    def projection_weight(self) -> torch.Tensor:
        return self.basis + torch.outer(self.normal, self.tilt)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.code(tokens), self.projection_weight())

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.projection_weight()
        latent = F.linear(x, weight.transpose(0, 1))
        return F.linear(latent, self.code.weight)


class GaugeFixedLayerNorm(nn.Module):
    """LayerNorm retaining six learned constant-shift directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 3:
            raise ValueError("normalized_shape must be at least three")

        self.norm = nn.LayerNorm(normalized_shape, bias=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 2))

        basis = torch.zeros(normalized_shape, normalized_shape - 2)
        for col in range(normalized_shape - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias_coeff @ self.basis.transpose(0, 1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
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
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = GaugeFixedLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
        self.pos_emb = QuotientPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
        self.final_bias = nn.Parameter(torch.zeros(cfg.d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        x = x + self.final_bias @ self.token_emb.basis.transpose(0, 1)
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE