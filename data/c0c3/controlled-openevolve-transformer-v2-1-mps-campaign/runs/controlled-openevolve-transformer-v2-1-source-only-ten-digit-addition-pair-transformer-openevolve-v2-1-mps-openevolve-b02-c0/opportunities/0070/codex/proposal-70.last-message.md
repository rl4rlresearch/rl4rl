MECHANISM: Incremental pre-MLP LayerNorm bias absorption

HYPOTHESIS: Fixing a sixth `ln2` bias coordinate to zero will reduce the model from 1,604 to 1,603 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the removed constant contribution.

INTENDED_EDIT: Store two rather than three `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final six coordinates as zeros.

EVIDENCE: The three preceding identical `ln2` bias reductions achieved 99.97%, 99.95%, and 99.97% accuracy at 1,606, 1,605, and 1,604 parameters, making one further coordinate anchoring the closest controlled test.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with five additive coordinates fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (x.size(-1),), self.weight, F.pad(self.bias, (0, 5)), 1e-5
        )
=======
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
>>>>>>> REPLACE