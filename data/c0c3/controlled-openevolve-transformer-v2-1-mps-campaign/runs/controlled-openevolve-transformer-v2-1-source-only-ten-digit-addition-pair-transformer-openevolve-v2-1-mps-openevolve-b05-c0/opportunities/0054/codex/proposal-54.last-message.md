MECHANISM: Coordinate-0 LayerNorm-bias absorption into the downstream MLP affine map

HYPOTHESIS: Fixing second-LayerNorm bias coordinate 0 at zero alongside coordinate 4 will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy because the full `fc1` bias can absorb both coordinates.

INTENDED_EDIT: Store six second-LayerNorm bias coordinates and reconstruct coordinates 0 and 4 as zero, retaining the successful scale-coordinate and MLP-output gauges.

EVIDENCE: The current coordinate-0 LayerNorm-scale gauge achieved 99.83%, and analogous value-bias coordinate 0 previously supported 99.94%; this makes coordinate 0 a better evidenced bias candidate than coordinate 1, whose LayerNorm-bias removal failed at 44.1%.

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
        full_bias = torch.cat(
            (
                self.bias[: self.fixed_index],
                self.bias.new_zeros(1),
                self.bias[self.fixed_index :],
            )
        )
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with selected scale and bias coordinates absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight.new_ones(2),
                self.weight[:2],
                self.weight.new_ones(1),
                self.weight[2:],
            )
        )
        full_bias = torch.cat(
            (
                self.bias.new_zeros(1),
                self.bias[:3],
                self.bias.new_zeros(1),
                self.bias[3:],
            )
        )
>>>>>>> REPLACE