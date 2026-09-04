MECHANISM: Vocabulary-translation gauge fixing

HYPOTHESIS: Anchoring one rank-seven token code to zero will reduce the verified 1,509-parameter model to 1,502 parameters while retaining at least 99% accuracy, because a shared translation of every token embedding can be absorbed by the quotient positional embeddings at input and becomes a class-independent logit shift at output.

INTENDED_EDIT: Apply the verified quotient-residual, `d_ff=11`, rank-seven token design, but learn codes for only 113 vocabulary entries and synthesize token zero’s code as a fixed zero vector.

EVIDENCE: The rank-seven factorized design reached 99.97% accuracy with 1,509 parameters; its token-code translation is an exact redundant degree of freedom when combined with zero-mean positional embeddings and softmax-invariant common logit shifts.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
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
    """Low-rank tied token map with its translation gauge fixed."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if not 0 < rank < embedding_dim:
            raise ValueError("rank must be between zero and embedding_dim")
        if num_embeddings < 2:
            raise ValueError("num_embeddings must be at least two")

        # The omitted first code is fixed at zero. Any unconstrained code
        # matrix can be translated into this gauge without changing the model
        # function, modulo quotient-position and common-logit shifts.
        self.code = nn.Embedding(num_embeddings - 1, rank)
        self.proj = nn.Linear(rank, embedding_dim, bias=False)

    def code_weight(self) -> torch.Tensor:
        return F.pad(self.code.weight, (0, 0, 1, 0))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        codes = F.embedding(tokens, self.code_weight())
        return self.proj(codes)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.proj.weight.transpose(0, 1))
        return F.linear(latent, self.code_weight())


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
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
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