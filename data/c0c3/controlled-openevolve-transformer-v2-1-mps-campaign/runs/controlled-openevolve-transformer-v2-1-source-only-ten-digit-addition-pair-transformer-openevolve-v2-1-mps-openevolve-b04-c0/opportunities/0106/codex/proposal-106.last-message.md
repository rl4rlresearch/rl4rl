MECHANISM: Fix the untested second MLP LayerNorm gain

HYPOTHESIS: Fixing MLP LayerNorm gain coordinate 2 at one will produce a 1574-parameter model with at least 99% accuracy, because coordinate 1 remains learned and shared with attention while coordinate 2 is zero-shift and absorbable into its downstream `fc1.weight` column.

INTENDED_EDIT: Store one MLP LayerNorm gain instead of two, reconstruct coordinate 2 as one, and retain the verified cross-sublayer sharing through coordinate 1.

EVIDENCE: Fixing coordinate 1 while leaving coordinate 2 learned collapsed to 15.25%, whereas keeping coordinate 1 learned through cross-sublayer sharing achieved 99.96% at 1575 parameters; this directly motivates preserving coordinate 1 and testing the unanchored coordinate 2.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with six fixed shifts and gains 3 through 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 6))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((ones(1), self.weight_rest, ones(5)))
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with only gain 1 learned and all other gains absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 7))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((ones(1), self.weight_rest, ones(6)))
>>>>>>> REPLACE