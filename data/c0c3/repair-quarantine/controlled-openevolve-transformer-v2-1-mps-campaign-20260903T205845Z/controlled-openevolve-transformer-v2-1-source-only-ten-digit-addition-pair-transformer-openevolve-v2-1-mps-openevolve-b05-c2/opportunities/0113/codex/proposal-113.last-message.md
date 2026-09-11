MECHANISM: Terminal final-normalization bias anchoring

HYPOTHESIS: Reproducing the qualified 1495-parameter design and fixing final LayerNorm bias coordinate 7 at zero while preserving coordinates 1–6 will yield 1494 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Reconstruct final LayerNorm bias coordinates 0 and 7 as zero, leaving coordinates 1–6 learned.

EVIDENCE: Fixing final bias coordinate 0 achieved 99.83% accuracy at 1495 parameters; additional anchors at coordinates 1–6 failed, with coordinate 2 reaching 98.03%. Coordinate 7 is the only remaining untested coordinate in this bias family and completes the coordinate-sensitivity probe.

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
class FirstAndLastBiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with final bias coordinates 0 and 7 fixed at zero."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (1, 1))
        return F.layer_norm(
            x, (self.dim,), self.weight, bias, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = FirstBiasAnchoredLayerNorm(cfg.d_model)
=======
        self.ln_f = FirstAndLastBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE