MECHANISM: Incremental pre-MLP LayerNorm scale absorption

HYPOTHESIS: Fixing a second `ln2` scale coordinate to one will reduce the model from 1,600 to 1,599 parameters while retaining at least 99% accuracy, because the unrestricted downstream `fc1` weight column can absorb the removed scaling degree.

INTENDED_EDIT: Store six learned `BiasAnchoredLayerNorm` scale coordinates and reconstruct the final two coordinates as fixed ones.

EVIDENCE: Fixing the first `ln2` scale coordinate to one retained 99.99% accuracy at 1,600 parameters; extending the identical constraint by one coordinate is the closest controlled test of this multiplicative redundancy.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with zero bias and its final scale fixed at one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None, 1e-5)
=======
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with zero bias and its final two scales fixed at one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 2), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None, 1e-5)
>>>>>>> REPLACE