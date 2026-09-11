MECHANISM: Coordinate-4 LayerNorm affine absorption into the downstream MLP

HYPOTHESIS: Fixing second-LayerNorm bias coordinate 4 at zero will reduce the model from 1620 to 1619 parameters while retaining at least 99% accuracy, because the full `fc1` bias can exactly absorb that coordinate.

INTENDED_EDIT: Store seven second-LayerNorm bias coordinates and reconstruct coordinate 4 as zero, alongside the already-fixed scale coordinate 4.

EVIDENCE: Coordinate 4 produced successful 1621-parameter MLP-bias and 1620-parameter LayerNorm-scale gauges, while the earlier unspecified LayerNorm-bias coordinate failed; the demonstrated coordinate sensitivity makes coordinate 4 the most informative bias candidate.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale coordinate 4 absorbed by the following affine map."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight[: self.fixed_index],
                self.weight.new_ones(1),
                self.weight[self.fixed_index :],
            )
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            full_weight,
            self.bias,
            self.eps,
        )
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale and bias coordinate 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight[: self.fixed_index],
                self.weight.new_ones(1),
                self.weight[self.fixed_index :],
            )
        )
        full_bias = torch.cat(
            (
                self.bias[: self.fixed_index],
                self.bias.new_zeros(1),
                self.bias[self.fixed_index :],
            )
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            full_weight,
            full_bias,
            self.eps,
        )
>>>>>>> REPLACE