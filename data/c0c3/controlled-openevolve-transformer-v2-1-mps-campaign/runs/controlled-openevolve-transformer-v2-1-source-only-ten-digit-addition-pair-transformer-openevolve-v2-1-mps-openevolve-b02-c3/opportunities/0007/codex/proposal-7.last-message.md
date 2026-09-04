MECHANISM: LayerNorm-quotient positional embeddings

HYPOTHESIS: Starting from the proven query-only-bias design, restricting each positional vector to the zero-mean feature subspace will remove `INPUT_LEN` additional parameters while maintaining at least 99% accuracy, because with zero dropout any omitted all-ones positional component is invisible through the pre-normalized residual stack and final LayerNorm.

INTENDED_EDIT: Apply query-only attention bias and represent positional embeddings with `d_model - 1` learned orthonormal-basis coefficients per position.

EVIDENCE: Query-only attention achieved 100% accuracy with 1628 parameters. Unlike the failed removals of query bias, LayerNorm affine terms, or MLP width, this change removes only a LayerNorm-invariant positional degree of freedom while preserving the model’s function class.

<<<<<<< SEARCH
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class QuotientPositionEmbedding(nn.Module):
    """Position embeddings modulo feature-wise all-ones shifts."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least two")

        self.coeff = nn.Embedding(num_embeddings, embedding_dim - 1)

        # Orthonormal basis for vectors whose feature-wise mean is zero.
        basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for col in range(embedding_dim - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.coeff(positions) @ self.basis.transpose(0, 1)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = QuotientPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE