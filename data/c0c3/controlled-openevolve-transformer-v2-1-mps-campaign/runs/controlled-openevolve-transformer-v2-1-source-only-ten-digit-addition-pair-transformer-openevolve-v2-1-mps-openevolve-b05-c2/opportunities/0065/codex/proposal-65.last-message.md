MECHANISM: Fifth pre-attention LayerNorm scale gauge

HYPOTHESIS: Fixing a fifth `ln1` scale will reduce the qualified model to 1514 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace the four-parameter `ln1` scale vector with three learned scales and five fixed unit scales.

EVIDENCE: The current four-scale-gauge design achieved 99.88% accuracy at 1515 parameters; the previous attempt at this exact reduction was malformed before verification, so a uniquely matching patch is the most informative next test.

<<<<<<< SEARCH
class QuadrupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with four scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 4), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
=======
class QuintupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with five scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 5), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = QuadrupleAnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = QuintupleAnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE