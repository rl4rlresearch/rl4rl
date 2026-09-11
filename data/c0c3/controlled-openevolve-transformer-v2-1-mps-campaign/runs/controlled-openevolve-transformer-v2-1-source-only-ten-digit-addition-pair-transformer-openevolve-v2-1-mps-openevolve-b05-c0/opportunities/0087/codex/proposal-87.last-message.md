MECHANISM: Fourth downstream-absorbable LayerNorm-scale gauge

HYPOTHESIS: Fixing LayerNorm-2 scale coordinate 2 will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because its bias is already fixed at zero and its scale can be absorbed exactly into the corresponding `fc1` input column.

INTENDED_EDIT: Remove LayerNorm-2 scale coordinate 2 and reconstruct it as one, extending the existing fixed scale set from coordinates 0, 1, and 4 to 0, 1, 2, and 4.

EVIDENCE: The 1607-parameter model achieved 99.97% accuracy while fixing three LayerNorm-2 scale coordinates, and the cited coordinate-1 LayerNorm-scale reduction independently reached 99.81%; coordinate 2 also has zero fixed bias, making it the cleanest remaining multiplicative downstream gauge.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 1, 2, 3, 4, 6 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 1, 2, 3, 4, 6)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))

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
    """LayerNorm with scale 0, 1, 2, 4 and bias 0, 1, 2, 3, 4, 6 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 1, 2, 3, 4, 6)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))

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