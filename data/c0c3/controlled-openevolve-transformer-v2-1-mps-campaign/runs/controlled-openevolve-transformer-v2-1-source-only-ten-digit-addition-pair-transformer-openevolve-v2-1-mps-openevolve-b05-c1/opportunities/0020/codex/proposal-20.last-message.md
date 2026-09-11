MECHANISM: Learned low-rank absolute-position manifold

HYPOTHESIS: Replacing independent eight-dimensional position vectors with arbitrary six-dimensional learned position codes and a shared learned projection will reduce parameters while retaining at least 99% accuracy after 40,000 steps.

INTENDED_EDIT: Replace the full positional table (`8L` parameters) with a rank-six factorization (`6L + 48` parameters), orthogonally initialized and variance-scaled, while preserving the successful attention and MLP widths.

EVIDENCE: The current 1,480-parameter model reached 100% accuracy, whereas fixed sinusoidal positions reached 0%; this suggests positional information is load-bearing but does not establish that every position needs an independent eight-dimensional vector. Free per-position codes preserve learned absolute addressing, and prior quotient-space models show that additional training can overcome slower convergence from constrained parameterizations.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


class LowRankPositionEmbedding(nn.Module):
    """Learned absolute-position codes decoded through a shared subspace."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if not 0 < rank <= embedding_dim:
            raise ValueError("rank must be between one and embedding_dim")
        self.codes = nn.Embedding(num_embeddings, rank)
        self.proj = nn.Linear(rank, embedding_dim, bias=False)
        self.scale = math.sqrt(embedding_dim / rank)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.proj(self.codes(positions)) * self.scale


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = LowRankPositionEmbedding(cfg.max_seq_len, cfg.d_model, rank=6)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.pos_emb.proj.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=28000)
=======
    p.add_argument("--train-steps", type=int, default=40000)
>>>>>>> REPLACE