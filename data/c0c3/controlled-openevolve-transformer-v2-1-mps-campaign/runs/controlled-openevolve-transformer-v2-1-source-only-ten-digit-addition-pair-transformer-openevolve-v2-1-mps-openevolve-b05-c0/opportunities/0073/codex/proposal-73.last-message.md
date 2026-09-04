MECHANISM: Third LayerNorm-bias/downstream-affine gauge

HYPOTHESIS: Fixing second-LayerNorm bias coordinate 3 will reduce the verified model from 1615 to 1614 parameters while retaining at least 99% accuracy, because its contribution remains absorbable by `fc1.bias`.

INTENDED_EDIT: Remove LayerNorm-2 bias coordinate 3 alongside coordinates 0 and 4, reconstructing all three as zeros during forward passes.

EVIDENCE: Coordinate 3 produced the closest simple 1615-parameter failure at 98.67% through the related LayerNorm-scale gauge, while fixing LayerNorm-2 bias coordinate 0 succeeded at 99.15%; this makes coordinate 3 the strongest evidence-backed next bias quotient.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 4)
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
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 3, 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 3, 4)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 3))

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
                self.bias[:2],
                self.bias.new_zeros(2),
                self.bias[2:],
            )
        )
>>>>>>> REPLACE