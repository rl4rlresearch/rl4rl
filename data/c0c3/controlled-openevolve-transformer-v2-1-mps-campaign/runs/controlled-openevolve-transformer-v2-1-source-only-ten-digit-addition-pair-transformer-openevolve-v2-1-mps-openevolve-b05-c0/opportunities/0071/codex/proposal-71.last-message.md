MECHANISM: Second LayerNorm-bias/downstream-affine gauge

HYPOTHESIS: Fixing second-LayerNorm bias coordinate 1 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because its contribution can be absorbed by the learned `fc1` bias.

INTENDED_EDIT: Remove LayerNorm-2 bias coordinate 1 alongside coordinate 4 and reconstruct both as zeros during forward passes.

EVIDENCE: The identical prior proposal was not evaluated because its search text matched multiple locations. Coordinate 1 remains the strongest untested choice: related value-bias and LayerNorm-scale reductions achieved 99.98% and 99.81%, while the verified 1616-parameter model already fixes LayerNorm-2 bias coordinate 4.

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
                self.bias[: self.fixed_index],
                self.bias.new_zeros(1),
                self.bias[self.fixed_index :],
            )
        )
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with selected scale and bias coordinates absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
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
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:3],
                self.bias.new_zeros(1),
                self.bias[3:],
            )
        )
>>>>>>> REPLACE