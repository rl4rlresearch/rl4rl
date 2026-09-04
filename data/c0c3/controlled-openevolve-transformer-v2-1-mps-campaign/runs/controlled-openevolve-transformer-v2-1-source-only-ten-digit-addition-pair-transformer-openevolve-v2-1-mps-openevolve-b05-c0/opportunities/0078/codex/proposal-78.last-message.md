MECHANISM: Triple virtual-AdamW LayerNorm-bias affine gauge

HYPOTHESIS: Virtually optimizing omitted LayerNorm-2 bias coordinate 5 alongside coordinates 1 and 6 will reduce the model from 1611 to 1610 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove LayerNorm-2 bias coordinate 5, reconstruct only coordinate 7 explicitly, and extend the downstream `fc1.bias` compensation and virtual optimizer state to coordinates 1, 5, and 6.

EVIDENCE: Dual virtual optimization of coordinates 1 and 6 achieved 99.98%, and virtual optimization rescued coordinate 6 from 73.2% to 99.86%. Coordinate 5 is the remaining non-anchor bias coordinate; its analogous value-bias pathway was empirically essential, motivating preserved virtual optimization instead of simple fixing.

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
        full_bias = torch.cat(
            (
                self.bias.new_zeros(5),
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
            )
        )
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0 through 6 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 1, 2, 3, 4, 5, 6)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 7))

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
                self.bias.new_zeros(7),
                self.bias,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.mlp.fc1.bias,
            (1, 6),
            (0, 2),
=======
            block.mlp.fc1.bias,
            (1, 5, 6),
            (0, 1, 2),
>>>>>>> REPLACE