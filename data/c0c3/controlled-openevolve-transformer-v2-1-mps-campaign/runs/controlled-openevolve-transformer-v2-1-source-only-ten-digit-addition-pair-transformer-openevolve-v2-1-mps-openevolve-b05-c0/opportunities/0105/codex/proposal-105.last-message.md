MECHANISM: Seventh LayerNorm-2 bias absorption gauge

HYPOTHESIS: Gauge-fixing LayerNorm-2 bias feature 7 will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because its effect is exactly absorbable into `fc1.bias` and the existing virtual-AdamW quotient already supports this gauge family.

INTENDED_EDIT: Remove LayerNorm-2 bias feature 7, retain feature 5, and extend the existing affine-bias optimizer mapping to reconstruct and optimize the omitted feature virtually.

EVIDENCE: The 1604-parameter model achieved 99.92% accuracy with six of eight LayerNorm-2 bias coordinates already fixed; extending that proven absorption mechanism avoids feature 5, which prior evidence identifies as optimization-sensitive.

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
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and all biases except 5 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 1, 2, 3, 4, 6, 7)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (
                self.bias.new_zeros(5),
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
            )
        )
=======
        full_bias = torch.cat(
            (
                self.bias.new_zeros(5),
                self.bias,
                self.bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            (1, 6),
            (0, 2),
=======
            (1, 6, 7),
            (0, 2, 3),
>>>>>>> REPLACE