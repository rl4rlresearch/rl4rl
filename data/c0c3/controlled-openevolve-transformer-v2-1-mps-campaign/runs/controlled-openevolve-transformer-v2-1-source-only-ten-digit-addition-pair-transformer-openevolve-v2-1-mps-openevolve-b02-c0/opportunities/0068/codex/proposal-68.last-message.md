MECHANISM: Incremental pre-MLP LayerNorm bias absorption

HYPOTHESIS: Fixing a fourth `ln2` bias coordinate to zero will reduce the model from 1,606 to 1,605 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the removed additive contribution.

INTENDED_EDIT: Store four rather than five `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final four coordinates as zeros.

EVIDENCE: Fixing the third `ln2` bias coordinate retained 99.97% accuracy at 1,606 parameters; extending the identical constraint by one coordinate is the closest controlled test of this successful redundancy.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with three additive coordinates fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (x.size(-1),), self.weight, F.pad(self.bias, (0, 3)), 1e-5
        )
=======
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
>>>>>>> REPLACE