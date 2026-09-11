MECHANISM: Incremental key-bias quotient with parameter-free RMS normalization

HYPOTHESIS: Removing a third softmax-invariant key-bias coordinate and training for 20,000 steps will retain at least 99% accuracy with 1,593 learned parameters.

INTENDED_EDIT: Replace all affine LayerNorms with parameter-free RMSNorm, reconstruct three fixed-zero key-bias coordinates, and extend training from 7,500 to 20,000 steps.

EVIDENCE: The 1,594-parameter design combining RMSNorm with two removed key-bias coordinates achieved 99.96% accuracy at 14,999 steps; the earlier progression from one to two removed coordinates showed that additional training recovers convergence after incremental redundant-coordinate removal.

<<<<<<< SEARCH
        # The first key-bias coordinate is softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 1))
=======
        # Three key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias.new_zeros(1),
=======
                self.qkv.bias.new_zeros(3),
>>>>>>> REPLACE

<<<<<<< SEARCH
class Block(nn.Module):
=======
class RMSNorm(nn.Module):
    """Parameter-free normalization that preserves the residual mean channel."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = RMSNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = RMSNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=7500)
=======
    p.add_argument("--train-steps", type=int, default=20000)
>>>>>>> REPLACE