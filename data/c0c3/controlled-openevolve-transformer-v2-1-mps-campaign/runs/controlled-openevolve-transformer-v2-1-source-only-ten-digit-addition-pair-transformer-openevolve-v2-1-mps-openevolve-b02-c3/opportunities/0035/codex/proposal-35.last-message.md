MECHANISM: Incremental pre-MLP bias gauge fixing

HYPOTHESIS: Reducing the qualified 1,043-parameter model’s pre-MLP LayerNorm bias from four learned directions to three will produce a 1,042-parameter transformer with at least 99% accuracy, because the removed fixed offset through `fc1` remains representable by `fc1.bias`.

INTENDED_EDIT: Apply the qualified five-coordinate lexical bottleneck, affine-free pre-attention LayerNorm, quotient-input MLP, and retain three learned zero-mean pre-MLP bias directions.

EVIDENCE: The 1,043-parameter four-direction design achieved 99.97% accuracy, while earlier incremental pre-MLP reductions from seven to six, six to five, and five to four all retained at least 99.95%; this makes one further direction the narrowest supported reduction.

<<<<<<< SEARCH
class FactorizedTokenEmbedding(nn.Module):
    """Gauge-fixed low-rank token map shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank != embedding_dim - 1:
            raise ValueError("rank must equal embedding_dim - 1")
=======
class FactorizedTokenEmbedding(nn.Module):
    """Fixed-subspace low-rank token map shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank < 1 or rank >= embedding_dim:
            raise ValueError("rank must be between one and embedding_dim - 1")
>>>>>>> REPLACE

<<<<<<< SEARCH
class BiasOnlyLayerNorm(nn.Module):
    """LayerNorm with fixed unit scale and a learned bias."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias
=======
class QuotientBiasLayerNorm(nn.Module):
    """Scale-fixed LayerNorm retaining three zero-mean bias directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 6:
            raise ValueError("normalized_shape must be at least six")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 5))

        basis = torch.zeros(normalized_shape, normalized_shape - 5)
        for col in range(normalized_shape - 5):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias_coeff @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = QuotientInputLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = GaugeFixedLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = BiasOnlyLayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = QuotientBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 3
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.final_bias = nn.Parameter(torch.zeros(cfg.d_model - 1))
=======
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim)
        )
>>>>>>> REPLACE