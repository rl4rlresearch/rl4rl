MECHANISM: Zero-initialized final-normalization bias anchoring

HYPOTHESIS: Fixing final LayerNorm bias coordinate 0 at zero will reduce the qualified model from 1496 to 1495 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Replace the final LayerNorm with an equivalent module whose scale and bias coordinates 1–7 remain learned while bias coordinate 0 is reconstructed as zero.

EVIDENCE: The current 1496-parameter design reached 99.96% accuracy; fixing final scale coordinate 0 failed, while earlier successful reductions anchored zero-initialized affine biases, motivating a direct test of the distinct final-normalization bias family.

<<<<<<< SEARCH
class FixedScaleLayerNorm(nn.Module):
    """LayerNorm with all scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), None, None, self.eps)


class Block(nn.Module):
=======
class FixedScaleLayerNorm(nn.Module):
    """LayerNorm with all scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), None, None, self.eps)


class FirstBiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with final bias coordinate 0 fixed at zero."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (1, 0))
        return F.layer_norm(
            x, (self.dim,), self.weight, bias, self.eps
        )


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = FirstBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE