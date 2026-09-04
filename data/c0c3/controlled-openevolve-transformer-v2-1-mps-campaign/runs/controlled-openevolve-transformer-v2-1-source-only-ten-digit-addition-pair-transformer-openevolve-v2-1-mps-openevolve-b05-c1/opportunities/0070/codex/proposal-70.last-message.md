MECHANISM: Full-rank sinusoidal positional subspace with learned orientation

HYPOTHESIS: Replacing the independent positional table with a normalized fixed eight-dimensional sinusoidal basis and a learned 8×8 projection will remove `8 * INPUT_LEN - 64` parameters while retaining at least 99% accuracy, because it preserves full positional rank and trainable orientation.

INTENDED_EDIT: Generate generic content-independent sinusoidal position codes, normalize them to match the existing embedding initialization scale after projection, and learn only their shared projection into the residual stream.

EVIDENCE: The learned rank-six positional factorization reached only 74.72%, identifying positional dimensionality as load-bearing, but it did not test whether independent vectors are necessary. This design restores all eight dimensions while challenging the assumption that every position needs separately learned parameters.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x[..., :-1])


class FixedBiasLayerNorm(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x[..., :-1])


class ProjectedSinusoidalPosition(nn.Module):
    """Full-rank fixed positional basis with a learned shared orientation."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        basis = torch.empty(max_seq_len, d_model)
        basis[:, 0::2] = torch.sin(position * frequency)
        basis[:, 1::2] = torch.cos(position * frequency)
        basis = basis / math.sqrt(d_model / 2)
        self.register_buffer("basis", basis, persistent=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.proj(self.basis[positions])


class FixedBiasLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = ProjectedSinusoidalPosition(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE