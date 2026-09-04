MECHANISM: Coordinate-6 LayerNorm-scale absorption into the downstream MLP affine map

HYPOTHESIS: Fixing second-LayerNorm scale coordinate 6 at one, alongside coordinates 0, 1, and 4, will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy because the full `fc1` weight can absorb all four scale degrees of freedom.

INTENDED_EDIT: Store four second-LayerNorm scale coordinates and reconstruct coordinates 0, 1, 4, and 6 as one, retaining the successful coordinate-4 bias and MLP-output gauges.

EVIDENCE: LayerNorm scale removals at coordinates 0, 1, and 4 jointly achieved 99.83%; coordinate 6 is the strongest untested scale candidate because the current successful model already fixes the analogous value-bias coordinate and LayerNorm-null key-projection row, without the negative MLP-bias evidence associated with coordinate 7.

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
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale coordinates 0, 1, 4, and 6 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_weight = torch.cat(
            (
                self.weight.new_ones(2),
                self.weight[:2],
                self.weight.new_ones(1),
                self.weight[2:],
            )
        )
=======
        full_weight = torch.cat(
            (
                self.weight.new_ones(2),
                self.weight[:2],
                self.weight.new_ones(1),
                self.weight[2:3],
                self.weight.new_ones(1),
                self.weight[3:],
            )
        )
>>>>>>> REPLACE