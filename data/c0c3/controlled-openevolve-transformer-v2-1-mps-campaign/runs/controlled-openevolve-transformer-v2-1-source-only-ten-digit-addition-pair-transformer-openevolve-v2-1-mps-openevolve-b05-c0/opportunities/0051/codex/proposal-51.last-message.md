MECHANISM: Coordinate-1 LayerNorm-scale absorption into the downstream MLP affine map

HYPOTHESIS: Fixing second-LayerNorm scale coordinate 1 at one, alongside coordinate 4, will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy because the full `fc1` weight can absorb both scale degrees of freedom.

INTENDED_EDIT: Store six second-LayerNorm scale coordinates and reconstruct coordinates 1 and 4 as one, while retaining the successful coordinate-4 bias gauge.

EVIDENCE: The additional coordinate-3 scale gauge narrowly missed at 98.67%, showing this quotient remains promising but coordinate-sensitive; analogous value-bias coordinate 1 achieved 99.98%, making coordinate 1 the strongest evidenced alternative.

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
>>>>>>> REPLACE