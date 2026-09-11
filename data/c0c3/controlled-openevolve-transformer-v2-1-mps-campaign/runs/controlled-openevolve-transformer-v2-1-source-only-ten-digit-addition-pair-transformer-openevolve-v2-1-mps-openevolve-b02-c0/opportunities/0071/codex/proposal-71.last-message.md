MECHANISM: Incremental pre-MLP LayerNorm bias absorption

HYPOTHESIS: Fixing a seventh `ln2` bias coordinate to zero will reduce the model from 1,603 to 1,602 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the removed constant contribution.

INTENDED_EDIT: Store one rather than two `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final seven coordinates as zeros.

EVIDENCE: The four most recent identical `ln2` bias reductions retained 99.95%–99.97% accuracy, including 99.97% at 1,603 parameters after fixing six coordinates; extending the same constraint by one coordinate is the closest controlled test.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with six additive coordinates fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (x.size(-1),), self.weight, F.pad(self.bias, (0, 6)), 1e-5
        )
=======
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with seven additive coordinates fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (x.size(-1),), self.weight, F.pad(self.bias, (0, 7)), 1e-5
        )
>>>>>>> REPLACE