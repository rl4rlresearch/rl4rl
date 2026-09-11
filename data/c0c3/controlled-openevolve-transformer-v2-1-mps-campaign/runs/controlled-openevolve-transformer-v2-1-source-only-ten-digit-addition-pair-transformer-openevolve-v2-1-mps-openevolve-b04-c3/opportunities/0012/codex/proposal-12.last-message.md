MECHANISM: Parameter-free RMS normalization combined with redundant key-bias elimination

HYPOTHESIS: Combining the proven two-coordinate key-bias elimination with parameter-free RMS normalization and training for 15,000 steps will achieve at least 99% accuracy with 1,594 parameters.

INTENDED_EDIT: Replace all three affine LayerNorms with parameter-free RMSNorm while retaining the current two-coordinate key-bias reduction, and extend training from 10,000 to 15,000 steps.

EVIDENCE: The all-RMSNorm design achieved 99.01% accuracy with 1,596 parameters at 10,000 steps, while the current two-coordinate key-bias reduction achieved 99.94%; additional training previously recovered reduced key-bias models from sub-threshold accuracy.

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
class RMSNorm(nn.Module):
    """Parameter-free normalization that preserves the residual mean channel."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = RMSNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=10000)
=======
    p.add_argument("--train-steps", type=int, default=15000)
>>>>>>> REPLACE