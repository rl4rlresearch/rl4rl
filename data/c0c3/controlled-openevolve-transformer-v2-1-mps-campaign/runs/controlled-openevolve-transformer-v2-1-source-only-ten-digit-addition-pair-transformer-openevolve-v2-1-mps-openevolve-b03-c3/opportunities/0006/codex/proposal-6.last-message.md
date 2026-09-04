MECHANISM: LayerNorm gauge-fixed positional embeddings

HYPOTHESIS: Starting from the verified 1,628-parameter design, removing each positional embedding’s all-ones component will preserve at least 99% accuracy while saving one parameter per sequence position, because that component is erased by the pre-block and final LayerNorms under the fixed zero-dropout configuration.

INTENDED_EDIT: Apply the verified key-bias and `ln2`-bias removals, then represent positional embeddings in a learned seven-dimensional orthonormal basis spanning the mean-zero subspace.

EVIDENCE: The 1,628-parameter design achieved 99.95% accuracy, whereas removing the full `ln1` bias failed at 33.96%; this motivates retaining optimization-sensitive offsets while eliminating an exact LayerNorm-invariant positional direction.

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


class MeanFreeEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, d_model: int):
        if d_model < 2:
            raise ValueError("d_model must be at least 2")
        super().__init__(num_embeddings, d_model - 1)

        eye = torch.eye(d_model)
        unit = torch.full((d_model,), 1.0 / math.sqrt(d_model))
        direction = eye[:, -1] - unit
        reflector = eye - 2.0 * torch.outer(direction, direction) / direction.dot(direction)
        self.register_buffer("basis", reflector[:, :-1], persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return super().forward(idx) @ self.basis.transpose(0, 1)


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
        self.v_bias = nn.Parameter(torch.zeros(d_model))
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
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanFreeEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE