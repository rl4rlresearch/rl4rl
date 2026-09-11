MECHANISM: Coordinate-1 LayerNorm bias absorption into the downstream MLP affine map

HYPOTHESIS: Fixing second-LayerNorm bias coordinate 1 at zero, alongside coordinate 4, will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy because the full `fc1` bias exactly absorbs both coordinates.

INTENDED_EDIT: Store six second-LayerNorm bias coordinates and reconstruct coordinates 1 and 4 as zero, while retaining the successful scale-coordinate and MLP-output gauges.

EVIDENCE: The coordinate-4 LayerNorm bias gauge achieved 99.84%, and coordinate 1 is the strongest additional candidate because its LayerNorm scale gauge achieved 99.81% and its analogous value-bias removal achieved 99.98%.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale coordinates 1 and 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:3],
                self.weight.new_ones(1),
                self.weight[3:],
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
    """LayerNorm with scale and bias coordinates 1 and 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:3],
                self.weight.new_ones(1),
                self.weight[3:],
            )
        )
        full_bias = torch.cat(
            (
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:3],
                self.bias.new_zeros(1),
                self.bias[3:],
            )
        )
>>>>>>> REPLACE