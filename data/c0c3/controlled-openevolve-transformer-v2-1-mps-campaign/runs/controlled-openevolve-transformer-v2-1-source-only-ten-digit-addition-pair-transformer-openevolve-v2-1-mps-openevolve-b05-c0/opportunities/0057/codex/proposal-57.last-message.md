MECHANISM: Coordinate-2 LayerNorm-scale absorption into the downstream MLP affine map

HYPOTHESIS: Fixing second-LayerNorm scale coordinate 2 at one, alongside coordinates 0, 1, and 4, will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy because the full `fc1` affine map can absorb all four scale degrees of freedom.

INTENDED_EDIT: Store four second-LayerNorm scale coordinates and reconstruct coordinates 0, 1, 2, and 4 as one, retaining the successful coordinate-4 LayerNorm-bias and MLP-output gauges.

EVIDENCE: The current three-coordinate scale gauge achieved 99.83%. Additional coordinates 3 and 6 were accuracy-sensitive, while the successful current model already removes analogous value-bias coordinate 2; this makes coordinate 2 the strongest remaining untested scale candidate.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale coordinates 0, 1, and 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight.new_ones(2),
                self.weight[:2],
                self.weight.new_ones(1),
                self.weight[2:],
            )
        )
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale coordinates 0, 1, 2, and 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight.new_ones(3),
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:],
            )
        )
>>>>>>> REPLACE