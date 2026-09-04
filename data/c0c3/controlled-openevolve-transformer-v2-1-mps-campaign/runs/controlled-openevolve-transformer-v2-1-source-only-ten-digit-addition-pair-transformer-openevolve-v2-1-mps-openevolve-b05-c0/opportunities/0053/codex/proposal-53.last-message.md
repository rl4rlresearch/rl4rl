MECHANISM: Coordinate-0 LayerNorm-scale absorption into the downstream MLP affine map

HYPOTHESIS: Fixing second-LayerNorm scale coordinate 0 at one, alongside coordinates 1 and 4, will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy because the full `fc1` weight can absorb all three scale degrees of freedom.

INTENDED_EDIT: Store five second-LayerNorm scale coordinates and reconstruct coordinates 0, 1, and 4 as one, retaining the successful coordinate-4 bias and MLP-output gauges.

EVIDENCE: The coordinate-1 scale gauge achieved 99.81%, while coordinate 3 narrowly missed at 98.67%, demonstrating coordinate sensitivity rather than exhaustion of the quotient. Analogous value-bias coordinate 0 later supported 99.94%, making coordinate 0 the strongest evidenced untested scale candidate.

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
=======
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
>>>>>>> REPLACE