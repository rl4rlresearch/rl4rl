MECHANISM: Incremental pre-MLP LayerNorm scale absorption

HYPOTHESIS: Fixing a fourth `ln2` scale coordinate to one will reduce the model from 1,598 to 1,597 parameters while retaining at least 99% accuracy, because the unrestricted downstream `fc1` weight column can absorb the removed scaling degree.

INTENDED_EDIT: Store four learned `BiasAnchoredLayerNorm` scale coordinates and reconstruct the final four coordinates as fixed ones.

EVIDENCE: Fixing the third `ln2` scale coordinate retained 99.94% accuracy at 1,598 parameters, after the preceding two identical reductions achieved 99.99% and 99.96%; extending the same constraint by one coordinate is the closest controlled test.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with zero bias and its final three scales fixed at one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 3), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None, 1e-5)
=======
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with zero bias and its final four scales fixed at one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 4), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None, 1e-5)
>>>>>>> REPLACE