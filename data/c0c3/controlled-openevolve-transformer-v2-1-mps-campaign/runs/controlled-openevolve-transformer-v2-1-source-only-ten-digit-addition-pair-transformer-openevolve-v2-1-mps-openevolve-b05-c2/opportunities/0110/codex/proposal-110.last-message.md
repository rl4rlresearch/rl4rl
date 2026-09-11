MECHANISM: Nonadjacent final-normalization bias coordinate 5 anchoring

HYPOTHESIS: Fixing final LayerNorm bias coordinate 5 while preserving coordinates 1–4 will reduce the qualified model from 1495 to 1494 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Reconstruct final LayerNorm bias coordinates 0 and 5 as zero, leaving coordinates 1–4 and 6–7 learned.

EVIDENCE: The 1495-parameter design reached 99.83% accuracy with coordinate 0 fixed; additional anchors at coordinates 1–4 failed, while the analogous MLP search succeeded specifically at nonadjacent coordinate 5 after lower coordinates failed, making coordinate 5 the most informative untested final-bias reduction.

<<<<<<< SEARCH
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
=======
class FirstAndCoordinateFiveBiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with final bias coordinates 0 and 5 fixed at zero."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (
                self.bias.new_zeros(1),
                self.bias[:4],
                self.bias.new_zeros(1),
                self.bias[4:],
            )
        )
        return F.layer_norm(
            x, (self.dim,), self.weight, bias, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = FirstBiasAnchoredLayerNorm(cfg.d_model)
=======
        self.ln_f = FirstAndCoordinateFiveBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE