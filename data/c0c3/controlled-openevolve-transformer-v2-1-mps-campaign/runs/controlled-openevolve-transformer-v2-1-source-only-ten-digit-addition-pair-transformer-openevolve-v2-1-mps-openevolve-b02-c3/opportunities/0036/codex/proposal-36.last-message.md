MECHANISM: Two-direction pre-MLP bias gauge fixing

HYPOTHESIS: Reducing the qualified 1,042-parameter model’s pre-MLP LayerNorm bias from three learned directions to two will produce a 1,041-parameter transformer with at least 99% accuracy, because the omitted constant contribution remains representable by `fc1.bias`.

INTENDED_EDIT: Apply the qualified five-coordinate lexical bottleneck, fixed sinusoidal positions, quotient-space attention and MLP maps, affine-free pre-attention normalization, and retain two learned zero-mean pre-MLP bias directions.

EVIDENCE: The 1,042-parameter three-direction design achieved 99.95% accuracy, and every preceding one-direction pre-MLP bias reduction retained at least 99.95%, supporting one further incremental reduction.

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


class QuotientInputLinear(nn.Module):
    """Linear map defined only on the zero-mean input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=bias)

        basis = torch.zeros(in_features, in_features - 1)
        for col in range(in_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(F.linear(x, self.basis.transpose(0, 1)))


class FactorizedTokenEmbedding(nn.Module):
    """Fixed-subspace low-rank token map shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank < 1 or rank >= embedding_dim:
            raise ValueError("rank must be between one and embedding_dim - 1")

        self.code = nn.Embedding(num_embeddings, rank)

        basis = torch.zeros(embedding_dim, rank)
        for col in range(rank):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.code(tokens), self.basis)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.basis.transpose(0, 1))
        return F.linear(latent, self.code.weight)


class QuotientBiasLayerNorm(nn.Module):
    """Scale-fixed LayerNorm retaining two zero-mean bias directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 7:
            raise ValueError("normalized_shape must be at least seven")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 6))

        basis = torch.zeros(normalized_shape, normalized_shape - 6)
        for col in range(normalized_shape - 6):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias_coeff @ self.basis.transpose(0, 1)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = QuotientInputLinear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
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
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        positions = torch.arange(seqlen, device=x.device)
        distance = positions[:, None] - positions[None, :]
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias[:, : seqlen - 1],
            ),
            dim=1,
        )
        att = att + relative_bias[:, distance.clamp_min(0)].unsqueeze(0)

        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc1 = QuotientInputLinear(d_model, d_ff)
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = QuotientBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
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
        self.cfg = cfg
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 3
        )

        positions = torch.arange(cfg.max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            / cfg.d_model
        )
        angles = positions * frequencies.unsqueeze(0)
        pos_encoding = torch.zeros(cfg.max_seq_len, cfg.d_model)
        pos_encoding[:, 0::2] = torch.sin(angles)
        odd_features = pos_encoding[:, 1::2].shape[1]
        pos_encoding[:, 1::2] = torch.cos(angles[:, :odd_features])
        pos_encoding = pos_encoding - pos_encoding.mean(dim=-1, keepdim=True)
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)
        self.pos_scale = nn.Parameter(torch.tensor(0.02))

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim)
        )

        self.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.token_emb(idx)
        x = x + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
        x = self.drop(x)
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