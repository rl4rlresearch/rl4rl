MECHANISM: Incremental pre-MLP LayerNorm bias absorption

HYPOTHESIS: Fixing a third `ln2` bias coordinate to zero will reduce the model from 1,607 to 1,606 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the removed additive contribution.

INTENDED_EDIT: Store five rather than six `BiasAnchoredLayerNorm` bias coordinates and reconstruct the final three coordinates as zeros.

EVIDENCE: Fixing the second `ln2` bias coordinate retained 99.67% accuracy at 1,607 parameters; extending the identical constraint by one coordinate is the closest controlled test of this successful redundancy.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE