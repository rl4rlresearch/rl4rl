MECHANISM: Fourth LayerNorm-bias/downstream-affine gauge

HYPOTHESIS: Fixing second-LayerNorm bias coordinate 2 will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because its constant downstream contribution can be absorbed by `fc1.bias`.

INTENDED_EDIT: Remove LayerNorm-2 bias coordinate 2 alongside coordinates 0, 3, and 4, reconstructing all four as zeros during forward passes.

EVIDENCE: The current three-coordinate bias quotient achieved 99.86% at 1614 parameters. Coordinate 2 is the most informative untested bias coordinate, while coordinate 1 already has direct negative evidence at 87.11%.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 3, 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 3, 4)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 3))
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 2, 3, 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 2, 3, 4)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (
                self.bias.new_zeros(1),
                self.bias[:2],
                self.bias.new_zeros(2),
                self.bias[2:],
            )
        )
=======
        full_bias = torch.cat(
            (
                self.bias.new_zeros(1),
                self.bias[:1],
                self.bias.new_zeros(3),
                self.bias[1:],
            )
        )
>>>>>>> REPLACE