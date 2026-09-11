MECHANISM: Incremental pre-MLP LayerNorm bias absorption

HYPOTHESIS: Fixing a second `ln2` bias coordinate to zero will reduce the model from 1,608 to 1,607 parameters while retaining at least 99% accuracy, because its effect can be absorbed by the unrestricted `fc1` bias.

INTENDED_EDIT: Store six rather than seven `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final two coordinates as zeros.

EVIDENCE: The 1,608-parameter design achieved 99.96% accuracy with one `ln2` bias coordinate already anchored; extending that constraint by one coordinate is a controlled reduction, and unlike the failed additional `fc2` column constraint, it precedes a linear layer with a learned bias that can absorb the removed additive degree.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with one additive coordinate fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (x.size(-1),), self.weight, F.pad(self.bias, (0, 1)), 1e-5
        )
=======
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with two additive coordinates fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (x.size(-1),), self.weight, F.pad(self.bias, (0, 2)), 1e-5
        )
>>>>>>> REPLACE