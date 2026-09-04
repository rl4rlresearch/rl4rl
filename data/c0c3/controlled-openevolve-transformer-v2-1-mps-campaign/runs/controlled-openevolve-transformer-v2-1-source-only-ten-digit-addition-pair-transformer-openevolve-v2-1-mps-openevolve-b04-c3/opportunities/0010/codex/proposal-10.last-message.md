MECHANISM: Parameter-free RMS pre-normalization

HYPOTHESIS: Replacing both affine pre-LayerNorms with parameter-free RMS normalization will preserve at least 99% accuracy after 10,000 training steps while reducing learned parameters from 1,644 to 1,612.

INTENDED_EDIT: Preserve the full-width embeddings, attention, MLP, and affine final LayerNorm, but normalize each residual branch by RMS without learned scale or bias and extend training to 10,000 steps.

EVIDENCE: Reducing `d_ff` removed an activation unit and collapsed accuracy to 74.93%, showing that representational width is load-bearing. RMS normalization instead removes 32 affine parameters while retaining the activation mean as an additional usable dimension; the 1,642-parameter reference also establishes that 10,000 steps can recover convergence in a perturbed parameterization.

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
    p.add_argument("--train-steps", type=int, default=5000)
=======
    p.add_argument("--train-steps", type=int, default=10000)
>>>>>>> REPLACE