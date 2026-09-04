MECHANISM: Head-specific learned relative-distance attention

HYPOTHESIS: Replacing independent seven-dimensional absolute position vectors with two learned causal distance-bias tables will retain at least 99% accuracy while reducing the verified 1,403-parameter design by `5 * INPUT_LEN` parameters, because each attention head can learn its own relative routing profile without compressing the independent query, key, or value maps.

INTENDED_EDIT: Adopt the verified three-quartet MLP and single attention-output bias tie, remove absolute position embeddings, and inject learned head-specific relative-distance biases directly into causal attention logits.

EVIDENCE: The 1,403-parameter attention-output-bias design achieved 99.98%, whereas sharing query/key projections fell to 93.49%; this preserves all content projections and instead challenges the load-bearing assumption that every absolute position requires a full learned residual-stream vector.

<<<<<<< SEARCH
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the feature-mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)
=======
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the feature-mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


class PairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with one adaptively shared bias pair."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 3:
            raise ValueError("out_features must be at least three")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.free_bias[:-1], self.free_bias[-1:].expand(2)))
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = MeanZeroLinear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        relative_distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        att = att + self.rel_bias[:, relative_distance].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with two learned-bias quartets and learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 8 or out_features % 2 != 0:
            raise ValueError("out_features must be even and at least eight")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 2 - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_biases = self.free_bias[:2].repeat_interleave(4)
        paired_biases = self.free_bias[2:].repeat_interleave(2)
        bias = torch.cat((quartet_biases, paired_biases))
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose outputs form learned-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = VocabCenteredEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.cfg = cfg
        self.token_emb = VocabCenteredEmbedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.token_emb(idx)
        x = self.drop(x)
>>>>>>> REPLACE