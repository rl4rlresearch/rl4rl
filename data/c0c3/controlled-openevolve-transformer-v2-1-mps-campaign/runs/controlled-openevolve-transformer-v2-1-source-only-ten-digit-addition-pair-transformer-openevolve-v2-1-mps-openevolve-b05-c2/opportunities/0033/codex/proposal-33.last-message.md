MECHANISM: Fifth-coordinate pre-attention scale-gauge anchoring

HYPOTHESIS: Fixing a fifth `ln1` scale coordinate at one will produce a 1538-parameter model with at least 99% accuracy, because the scale is absorbable into the corresponding unrestricted `qkv` input column while three learned scales remain.

INTENDED_EDIT: Extend the qualified four-coordinate `ln1` scale anchor by one coordinate without changing initialization or any other model component.

EVIDENCE: Four anchored scales achieved 99.95% accuracy at 1539 parameters after each of the preceding one-, two-, and three-coordinate anchors also qualified; the failure from removing all eight scales motivates continuing one coordinate at a time.

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