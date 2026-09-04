MECHANISM: Sixth pre-MLP bias gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm bias coordinate 6 at zero will reduce the model from 1610 to 1609 parameters while retaining at least 99% accuracy, because its shift is absorbable by `fc1.bias`, its gain is already fixed, and unlike remaining bias coordinates 2 and 5 it has not caused a prior failure.

INTENDED_EDIT: Remove bias coordinate 6 from the learned vector and reconstruct bias coordinates 0, 1, 3, 4, 6, and 7 as fixed zeros.

EVIDENCE: The 1610-parameter design reached 99.91% after fixing bias coordinate 4; coordinates 2 and 5 previously collapsed when fixed, whereas coordinate 6’s distributed embedding anchor retained 99.72% and its gain was fixed in the 100%-accurate 1611-parameter design.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with five fixed shifts and gains 3 through 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 5))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((self.weight_rest, ones(5)))
        bias = torch.cat(
            (zeros(2), self.bias_rest[:1], zeros(2), self.bias_rest[1:], zeros(1))
        )
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with six fixed shifts and gains 3 through 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 5))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((self.weight_rest, ones(5)))
        bias = torch.cat(
            (zeros(2), self.bias_rest[:1], zeros(2), self.bias_rest[1:], zeros(2))
        )
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE