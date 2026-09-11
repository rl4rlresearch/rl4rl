MECHANISM: Alternative attention-input gain anchoring

HYPOTHESIS: Fixing `ln1.weight[0]` at one will reduce the model from 1592 to 1591 parameters while retaining at least 99% accuracy, because coordinate 0 may be optimization-compatible even though fixing coordinate 3—whose shift is already anchored—failed.

INTENDED_EDIT: Replace the eight-parameter first LayerNorm gain with seven learned coordinates, reconstructing gain coordinate 0 as one while continuing to fix shift coordinate 3.

EVIDENCE: Fixing `ln1.weight[3]` reached only 74.01%, but the final-MLP bias experiments demonstrated strong coordinate sensitivity: fixing `fc2.bias[0]` reached 2.79%, whereas fixing `fc2.bias[1]` achieved 99.93% at 1592 parameters.

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
        zero = self.bias_rest.new_zeros(1)
        one = self.weight_rest.new_ones(1)
        weight = torch.cat((one, self.weight_rest))
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE