MECHANISM: Fifth pre-MLP bias gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm bias coordinate 4 at zero will reduce the model from 1611 to 1610 parameters while retaining at least 99% accuracy, because its shift is absorbable by `fc1.bias` and coordinate 4 has remained viable while its gain was fixed.

INTENDED_EDIT: Remove bias coordinate 4 from the learned vector and reconstruct bias coordinates 0, 1, 3, 4, and 7 as fixed zeros.

EVIDENCE: The current design achieved 100% accuracy at 1611 parameters with gain coordinate 4 fixed, and coordinate 4’s distributed embedding anchor previously retained 99.94%; this makes it the strongest untested remaining bias coordinate after bias coordinates 2 and 5 caused failures.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with four fixed shifts and gains 3 through 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 5))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((self.weight_rest, ones(5)))
        bias = torch.cat(
            (zeros(2), self.bias_rest[:1], zeros(1), self.bias_rest[1:], zeros(1))
        )
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
=======
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
>>>>>>> REPLACE