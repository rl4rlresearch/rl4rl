MECHANISM: Incremental pre-MLP LayerNorm bias absorption

HYPOTHESIS: Fixing a fifth `ln2` bias coordinate to zero will reduce the model from 1,605 to 1,604 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb its constant additive contribution.

INTENDED_EDIT: Store three rather than four `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final five coordinates as zeros.

EVIDENCE: Fixing the fourth `ln2` bias coordinate retained 99.95% accuracy at 1,605 parameters, after the two preceding identical reductions also exceeded 99%; extending the same constraint by one coordinate is the closest controlled test.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with four additive coordinates fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (x.size(-1),), self.weight, F.pad(self.bias, (0, 4)), 1e-5
        )
=======
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
>>>>>>> REPLACE