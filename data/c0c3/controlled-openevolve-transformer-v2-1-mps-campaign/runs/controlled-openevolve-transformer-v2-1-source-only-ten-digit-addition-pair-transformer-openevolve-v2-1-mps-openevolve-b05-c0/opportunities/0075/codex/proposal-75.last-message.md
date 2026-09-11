MECHANISM: Fifth LayerNorm-bias/downstream-affine gauge at coordinate 6

HYPOTHESIS: Fixing second-LayerNorm bias coordinate 6 will reduce the model from 1613 to 1612 parameters while retaining at least 99% accuracy, because its constant contribution can be absorbed by `fc1.bias`.

INTENDED_EDIT: Remove LayerNorm-2 bias coordinate 6 alongside coordinates 0, 2, 3, and 4, reconstructing all five as zeros during forward passes.

EVIDENCE: Successive LayerNorm-2 bias reductions reached 99.96% at 1613 parameters; coordinate 6 is an untested retained coordinate that avoids feature 5’s empirically essential value-bias pathway and feature 7’s use as the embedding and QKV gauge anchor.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 2, 3, 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 2, 3, 4)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 4))

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
                self.bias[:1],
                self.bias.new_zeros(3),
                self.bias[1:],
            )
        )
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 2, 3, 4, 6 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 2, 3, 4, 6)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 5))

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
                self.bias[:1],
                self.bias.new_zeros(3),
                self.bias[1:2],
                self.bias.new_zeros(1),
                self.bias[2:],
            )
        )
>>>>>>> REPLACE