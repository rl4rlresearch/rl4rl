MECHANISM: Pre-MLP LayerNorm scale absorption

HYPOTHESIS: The qualified 1,325-parameter relative-position transformer will retain at least 99% accuracy with 1,317 parameters because the pre-MLP LayerNorm scale can be absorbed exactly into `fc1`’s unrestricted input columns, while retaining the LayerNorm bias that prior evidence indicates is optimization-critical.

INTENDED_EDIT: Apply the qualified 1,325-parameter fixed-token-basis, sinusoidal/relative-distance architecture and replace `ln2` with a scale-fixed LayerNorm that retains all eight learned bias parameters.

EVIDENCE: The 1,325-parameter reference achieved 99.97% accuracy, and fixing the analogous pre-attention LayerNorm scale retained 99.96% accuracy. Unlike the unsuccessful removal of `ln2` bias, this patch preserves that bias and removes only a scale exactly reparameterizable through the following learned affine layer.

<<<<<<< SEARCH
        normal = torch.ones(embedding_dim) / math.sqrt(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("normal", normal, persistent=False)
        self.tilt = nn.Parameter(torch.zeros(rank))
=======
        self.register_buffer("basis", basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def projection_weight(self) -> torch.Tensor:
        return self.basis + torch.outer(self.normal, self.tilt)
=======
    def projection_weight(self) -> torch.Tensor:
        return self.basis
>>>>>>> REPLACE

<<<<<<< SEARCH
class QuotientLayerNorm(nn.Module):
    """LayerNorm whose bias omits its attention-redundant all-ones direction."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, bias=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 1))

        basis = torch.zeros(normalized_shape, normalized_shape - 1)
        for col in range(normalized_shape - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias_coeff @ self.basis.transpose(0, 1)
=======
class GaugeFixedLayerNorm(nn.Module):
    """Scale-fixed LayerNorm retaining five learned attention-bias directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 4:
            raise ValueError("normalized_shape must be at least four")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 3))

        basis = torch.zeros(normalized_shape, normalized_shape - 3)
        for col in range(normalized_shape - 3):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias_coeff @ self.basis.transpose(0, 1)


class BiasOnlyLayerNorm(nn.Module):
    """LayerNorm with fixed unit scale and a learned bias."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
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
        self.ln1 = QuotientLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = GaugeFixedLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = BiasOnlyLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
        self.pos_emb = QuotientPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
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