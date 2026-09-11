MECHANISM: Fourth LayerNorm-scale absorption gauge

HYPOTHESIS: Fixing LayerNorm-2 scale coordinate 2 will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because that scale can be absorbed into the downstream `fc1` weight column and three coordinates already train successfully under the same gauge.

INTENDED_EDIT: Remove LayerNorm-2 scale coordinate 2 by reconstructing it as one, leaving trainable scales only at features 3, 5, 6, and 7.

EVIDENCE: The 1604-parameter design achieved 99.92% accuracy with LayerNorm-2 scales 0, 1, and 4 already fixed; extending this proven gauge family avoids another attention-output coordinate after three fourth-coordinate trials timed out.

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