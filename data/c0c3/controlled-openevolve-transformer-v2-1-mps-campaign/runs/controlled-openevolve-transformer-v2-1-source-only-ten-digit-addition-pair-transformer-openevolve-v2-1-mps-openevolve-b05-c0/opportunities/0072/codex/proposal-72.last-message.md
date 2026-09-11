MECHANISM: Second LayerNorm-bias/downstream-affine gauge at coordinate 0

HYPOTHESIS: Fixing second-LayerNorm bias coordinate 0 will reduce the model to 1615 parameters while retaining at least 99% accuracy, because its contribution is absorbable by `fc1.bias` and coordinate 0 previously tolerated two related reductions above the accuracy threshold.

INTENDED_EDIT: Remove LayerNorm-2 bias coordinate 0 alongside the existing fixed coordinate 4 and reconstruct both as zeros during forward passes.

EVIDENCE: Coordinate 0 achieved 99.83% under the related LayerNorm-scale gauge and 99.94% under the analogous value-bias reduction; this makes it the strongest evidence-backed alternative after coordinate-1 LayerNorm-bias fixing reached only 87.11%.

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
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 4 absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_indices = (0, 4)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        return F.layer_norm(
=======
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
        return F.layer_norm(
>>>>>>> REPLACE