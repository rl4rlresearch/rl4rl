MECHANISM: Coordinate-3 LayerNorm-scale absorption into the downstream MLP affine map

HYPOTHESIS: Fixing second-LayerNorm scale coordinate 3 at one will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because the full `fc1` weight can exactly absorb this additional scale degree of freedom.

INTENDED_EDIT: Store six second-LayerNorm scale coordinates, reconstruct coordinates 3 and 4 as one, and retain the existing coordinate-4 bias gauge.

EVIDENCE: Fixing scale coordinate 4 reached 99.91%, and subsequently fixing bias coordinate 4 retained 99.84%; coordinate 3 is the other explicitly reported coordinate whose analogous value-bias removal eventually succeeded, making it the strongest next coordinate-specific test.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale and bias coordinate 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight[: self.fixed_index],
                self.weight.new_ones(1),
                self.weight[self.fixed_index :],
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
    """LayerNorm with scale coordinates 3/4 and bias coordinate 4 absorbed."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_scale_index = 3
        self.fixed_bias_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight[: self.fixed_scale_index],
                self.weight.new_ones(2),
                self.weight[self.fixed_scale_index :],
            )
        )
        full_bias = torch.cat(
            (
                self.bias[: self.fixed_bias_index],
                self.bias.new_zeros(1),
                self.bias[self.fixed_bias_index :],
            )
        )
>>>>>>> REPLACE