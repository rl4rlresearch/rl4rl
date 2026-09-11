MECHANISM: Distributed pre-attention gain gauge fixing

HYPOTHESIS: Fixing pre-attention LayerNorm gain coordinate 0 at one will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because unlike the 98.82%-accurate coordinate-3 attempt, it does not compound the existing fixed shift on the same coordinate.

INTENDED_EDIT: Replace the eight learned pre-attention LayerNorm gains with seven learned gains, reconstructing gain coordinate 0 as one while leaving the existing coordinate-3 shift anchor unchanged.

EVIDENCE: Fixing gain coordinate 3 reached 98.82%, close to the threshold, showing the gain redundancy is nearly viable; distributing the gain and shift constraints across different coordinates directly tests whether the remaining loss was caused by coordinate-3’s compounded affine anchoring.

<<<<<<< SEARCH
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with shift coordinate 3 absorbed by attention biases."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gain coordinate 0 and shift coordinate 3 fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 1))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one = self.weight_rest.new_ones(1)
        zero = self.bias_rest.new_zeros(1)
        weight = torch.cat((one, self.weight_rest))
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE