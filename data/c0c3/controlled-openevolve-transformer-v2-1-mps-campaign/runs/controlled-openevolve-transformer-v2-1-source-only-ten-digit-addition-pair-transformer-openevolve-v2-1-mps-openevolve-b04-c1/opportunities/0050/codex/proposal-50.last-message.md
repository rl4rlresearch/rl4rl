MECHANISM: Shared multiscale Fourier positional representation

HYPOTHESIS: Replacing the independent 8-parameter lookup at every position with generic multiscale Fourier features and four learned frequency gains will retain at least 99% accuracy while reducing parameters by `8 * max_seq_len - 4`, because attention needs a consistent positional geometry more than unrelated vectors for every position.

INTENDED_EDIT: Preserve the verified transformer and MLP widths, but replace the learned absolute-position table with a compact shared Fourier encoder; preserve downstream seeded initialization draws for a clean comparison.

EVIDENCE: Narrowing the MLP and adding recurrent refinement collapsed to 24.97%, while width-preserving reductions repeatedly retained at least 99%; this motivates preserving load-bearing channel capacity and challenging the shared assumption that every sequence position needs an independent learned vector.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())


class CausalSelfAttention(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())


class FourierPositionEmbedding(nn.Module):
    """Shared generic position code with one learned gain per frequency."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("Fourier position encoding requires an even d_model")

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        n_frequency = d_model // 2

        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        log_wavelengths = torch.linspace(
            math.log(4.0), math.log(2.0 * max_seq_len), n_frequency
        )
        frequencies = 2.0 * math.pi * torch.exp(-log_wavelengths)
        angles = positions * frequencies.unsqueeze(0)
        basis = torch.stack((angles.sin(), angles.cos()), dim=-1).reshape(
            max_seq_len, d_model
        )
        self.register_buffer("basis", basis, persistent=False)
        self.scale = nn.Parameter(
            torch.full((n_frequency,), 0.02 * math.sqrt(2.0))
        )

        # Match nn.Embedding's constructor draw so subsequent modules retain
        # the verified design's seeded initialization stream.
        initialization_draw = torch.empty(max_seq_len, d_model)
        nn.init.normal_(initialization_draw)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.scale.fill_(0.02 * math.sqrt(2.0))

        # Match the dense positional table's initialization draw during apply.
        initialization_draw = self.scale.new_empty(
            self.max_seq_len, self.d_model
        )
        nn.init.normal_(initialization_draw, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        scale = self.scale.repeat_interleave(2)
        return F.embedding(idx, self.basis) * scale


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = ShiftGaugeEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = ShiftGaugeEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = FourierPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
        elif isinstance(module, OneColumnGaugeLinear):
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, FourierPositionEmbedding):
            module.reset_parameters()
        elif isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
        elif isinstance(module, OneColumnGaugeLinear):
>>>>>>> REPLACE