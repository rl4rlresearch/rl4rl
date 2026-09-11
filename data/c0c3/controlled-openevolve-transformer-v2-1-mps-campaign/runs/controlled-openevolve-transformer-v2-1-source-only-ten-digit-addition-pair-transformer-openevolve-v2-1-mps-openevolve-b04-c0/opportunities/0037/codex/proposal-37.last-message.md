MECHANISM: Bias-sensitive coordinate scale gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm gain coordinate 2 to one will reduce the model from 1611 to 1610 parameters while retaining at least 99% accuracy, because its scale is absorbable by `fc1.weight` while its known-sensitive bias remains learned.

INTENDED_EDIT: Remove gain coordinate 2 from the learned vector and reconstruct gains 2 through 7 as fixed ones.

EVIDENCE: Coordinate 5 bias fixing collapsed to 73.24%, but fixing its gain retained 99.95%, showing that bias sensitivity does not imply gain sensitivity; coordinate 2 can therefore retain its learned shift while testing the same successful scale redundancy.

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
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with four fixed shifts and gains 2 through 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 6))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((self.weight_rest, ones(6)))
>>>>>>> REPLACE