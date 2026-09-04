MECHANISM: Nonadjacent final-normalization bias coordinate 4 anchoring

HYPOTHESIS: Fixing final LayerNorm bias coordinate 4 while preserving coordinates 1–3 will reduce the qualified 1495-parameter model to 1494 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Reconstruct final LayerNorm bias coordinates 0 and 4 as zero, leaving coordinates 1–3 and 5–7 learned.

EVIDENCE: Fixing coordinate 0 achieved 99.83% accuracy; additional anchors at coordinates 1 and 3 failed substantially, while coordinate 2 reached 98.03%. Coordinate 4 is the next untested coordinate and extends the coordinate-sensitivity probe without removing any previously implicated coordinate.

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
    """LayerNorm with final bias coordinates 0 and 4 fixed at zero."""
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
                self.bias[:3],
                self.bias.new_zeros(1),
                self.bias[3:],
            )
        )
        return F.layer_norm(
            x, (self.dim,), self.weight, bias, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = FirstBiasAnchoredLayerNorm(cfg.d_model)
=======
        self.ln_f = FirstAndFifthBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE