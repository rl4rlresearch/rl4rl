MECHANISM: Incremental pre-MLP LayerNorm scale absorption

HYPOTHESIS: Fixing a fifth `ln2` scale coordinate to one will reduce the model from 1,597 to 1,596 parameters while retaining at least 99% accuracy, because the corresponding unrestricted downstream `fc1` weight column can absorb the removed scaling degree.

INTENDED_EDIT: Store three learned `BiasAnchoredLayerNorm` scale coordinates and reconstruct the final five coordinates as fixed ones.

EVIDENCE: Fixing the fourth `ln2` scale coordinate retained 99.27% accuracy at 1,597 parameters, while the three preceding identical reductions retained 99.94%–99.99%; one further coordinate is the closest controlled test of this established redundancy.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with zero bias and its final four scales fixed at one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 4), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None, 1e-5)
=======
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with zero bias and its final five scales fixed at one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 5), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None, 1e-5)
>>>>>>> REPLACE