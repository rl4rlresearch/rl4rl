MECHANISM: Incremental pre-MLP LayerNorm scale absorption

HYPOTHESIS: Fixing one `ln2` scale coordinate to one will reduce the model from 1,601 to 1,600 parameters while retaining at least 99% accuracy, because the unrestricted downstream `fc1` weight column can absorb that coordinate’s learned scaling.

INTENDED_EDIT: Store seven learned `BiasAnchoredLayerNorm` scale coordinates and reconstruct the eighth as a fixed one.

EVIDENCE: Removing all eight `ln2` bias coordinates retained 99.96% accuracy at 1,601 parameters; this tests the next affine redundancy incrementally, with the same downstream `fc1` providing exact multiplicative absorption through its corresponding weight column.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with its additive bias fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), self.weight, None, 1e-5)
=======
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with zero bias and its final scale fixed at one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None, 1e-5)
>>>>>>> REPLACE