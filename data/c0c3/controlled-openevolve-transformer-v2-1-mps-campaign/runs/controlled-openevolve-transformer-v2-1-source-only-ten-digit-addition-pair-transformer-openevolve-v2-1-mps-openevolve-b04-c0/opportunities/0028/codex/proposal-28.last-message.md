MECHANISM: Nonconsecutive pre-MLP LayerNorm gauge fixing

HYPOTHESIS: Fixing the final pre-MLP LayerNorm bias coordinate instead of coordinate 2 will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because the shift remains exactly absorbable by `fc1.bias` and avoids the coordinate whose removal caused the 48.74% collapse.

INTENDED_EDIT: Keep bias coordinates 0 and 1 fixed, additionally fix coordinate 7, and retain learned biases for coordinates 2 through 6.

EVIDENCE: Fixing coordinates 0 and 1 achieved 99.92% at 1618 parameters, whereas also fixing coordinate 2 fell to 48.74%. Coordinate 2 is the first detached positional anchor concentrated on token row 0, while coordinate 7 uses the successfully distributed token-row-5 endpoint, motivating a coordinate-specific rather than cumulative test.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with two shift coordinates absorbed by the following MLP bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias_rest.new_zeros(2), self.bias_rest))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with the first two and final shifts absorbed by the MLP bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        bias = torch.cat((zeros(2), self.bias_rest, zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE