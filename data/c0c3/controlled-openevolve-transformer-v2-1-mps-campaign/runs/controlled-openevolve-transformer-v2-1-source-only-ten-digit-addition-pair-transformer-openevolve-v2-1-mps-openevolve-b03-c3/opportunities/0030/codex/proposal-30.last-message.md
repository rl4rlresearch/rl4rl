MECHANISM: Learned relative-lag attention routing

HYPOTHESIS: Replacing full-width absolute position embeddings with a learned two-head relative-lag bias will achieve at least 99% accuracy with 1,488 parameters, because addition’s repeated sequence relationships can be routed by learned relative offsets while retaining the load-bearing lexical, query, value, and MLP capacity.

INTENDED_EDIT: Remove the 183-parameter absolute positional representation and inject a 46-parameter learned relative-distance bias directly into causal attention, preserving the established initialization stream and existing successful gauge optimizers.

EVIDENCE: The 1,625-parameter current model reached 99.93%, while rank-seven token factorization collapsed to 3.76% and query-bias removal collapsed to 48.92%. This identifies the lexical interface and content-based attention as load-bearing, motivating a different reduction of positional representation rather than another coordinate gauge or embedding-rank reduction.

<<<<<<< SEARCH
class GaugeFixedPositionEmbedding(nn.Module):
    """Embedding with one shift-invariant positional scalar removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first = nn.Parameter(torch.empty(embedding_dim - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings - 1, embedding_dim))
        self.full_first = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        # Draw the original full tensor so initialization and downstream RNG
        # consumption match nn.Embedding exactly.
        raw = self.first.new_empty(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(raw, mean=0.0, std=std)
        self.first.copy_(raw[0, :-1] - raw[0, -1])
        self.rest.copy_(raw[1:])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            self.full_first = first
        weight = torch.cat((first.unsqueeze(0), self.rest), dim=0)
        return F.embedding(idx, weight)
=======
class LearnedRelativePositionBias(nn.Module):
    """Shift-equivariant positional routing learned in attention-logit space."""

    def __init__(self, n_head: int, max_seq_len: int, rng_width: int):
        super().__init__()
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.rng_width = rng_width
        self.bias = nn.Parameter(torch.empty(n_head, max_seq_len))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        # Draw the former full positional tensor to preserve the initialization
        # stream of every unchanged transformer parameter.
        raw = self.bias.new_empty(self.max_seq_len, self.rng_width)
        nn.init.normal_(raw, mean=0.0, std=std)
        self.bias.copy_(
            raw.flatten()[: self.bias.numel()].view_as(self.bias)
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        positions = torch.arange(seqlen, device=self.bias.device)
        distance = (
            positions[:, None] - positions[None, :]
        ).clamp_min(0)
        return self.bias[:, distance]
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
=======
    def forward(
        self, x: torch.Tensor, position_bias: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att + position_bias.unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(
        self, x: torch.Tensor, position_bias: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), position_bias)
        x = x + self.mlp(self.ln2(x))
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_bias = LearnedRelativePositionBias(
            cfg.n_head, cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
=======
        if isinstance(module, LearnedRelativePositionBias):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedBiasLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)
=======
        x = self.drop(self.token_emb(idx))
        position_bias = self.pos_bias(seqlen)

        for blk in self.blocks:
            x = blk(x, position_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Optimize each seven-coordinate gauge parameter through the AdamW
    # moments of its corresponding full eight-coordinate vector.
    gauge_params = [model.pos_emb.first] + [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
=======
    # Optimize each seven-coordinate MLP-bias gauge through the AdamW
    # moments of its corresponding full eight-coordinate vector.
    gauge_params = [
        blk.mlp.fc2.bias for blk in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()] + [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
=======
        full_gauge_grads = [
            blk.mlp.fc2.full_bias.grad.detach() for blk in model.blocks
        ]
>>>>>>> REPLACE