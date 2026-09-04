MECHANISM: Nonadjacent final-normalization bias coordinate 5 anchoring

HYPOTHESIS: Fixing final LayerNorm bias coordinate 5 while preserving coordinates 1–4 and 6–7 will reduce the qualified 1495-parameter model to 1494 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reconstruct final LayerNorm bias coordinates 0 and 5 as zero, leaving the other six bias coordinates and all eight scales learned.

EVIDENCE: The current design achieved 99.83% accuracy with coordinate 0 fixed; additional anchors at coordinates 1–4 have been tested, with coordinate 2 reaching 98.03%. Coordinate 5 is the next untested member of this successful zero-initialized bias family.

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
class FirstAndFifthBiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with final bias coordinates 0 and 5 fixed at zero."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        remaining_bias = torch.cat(
            (
                self.bias[:4],
                self.bias.new_zeros(1),
                self.bias[4:],
            )
        )
        bias = F.pad(remaining_bias, (1, 0))
        return F.layer_norm(
            x, (self.dim,), self.weight, bias, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = FirstBiasAnchoredLayerNorm(cfg.d_model)
=======
        self.ln_f = FirstAndFifthBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE