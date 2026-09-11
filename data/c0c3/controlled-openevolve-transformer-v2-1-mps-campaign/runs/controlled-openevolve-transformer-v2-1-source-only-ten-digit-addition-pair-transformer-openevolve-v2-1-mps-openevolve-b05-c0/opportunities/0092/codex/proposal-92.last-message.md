MECHANISM: Virtual-AdamW quotient of LayerNorm-2 bias feature 7

HYPOTHESIS: Gauge-fixing LayerNorm-2 bias coordinate 7 and absorbing its virtual AdamW updates into `fc1.bias` will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove `ln2.bias[7]`, reconstruct it as zero, and extend the existing affine-bias virtual optimizer and gradient-clipping machinery to preserve its gradient, moments, and downstream effect.

EVIDENCE: Virtual optimization previously rescued an omitted LayerNorm-bias coordinate to 99.86%, and the current 1607-parameter design already fixes six LayerNorm-2 bias coordinates. Fixing feature 7 extends that proven gauge while retaining feature 5, the feature index with the strongest evidence of bias-pathway sensitivity.

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
    """LayerNorm with scale 0, 1, 4 and bias 0, 1, 2, 3, 4, 6, 7 absorbed downstream."""

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
            block.mlp.fc1.bias,
            (1, 6),
            (0, 2),
=======
            block.mlp.fc1.bias,
            (1, 6, 7),
            (0, 2, 3),
>>>>>>> REPLACE