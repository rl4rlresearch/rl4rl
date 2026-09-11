MECHANISM: Attention-input gain gauge anchoring

HYPOTHESIS: Fixing `ln1.weight[3]` at one will reduce the model from 1592 to 1591 parameters while retaining at least 99% accuracy, because its scale can be absorbed into column 3 of the learned QKV projection.

INTENDED_EDIT: Replace the eight-parameter first LayerNorm gain with seven learned coordinates, reconstructing gain coordinate 3 as one while retaining the existing fixed shift at coordinate 3.

EVIDENCE: The 1592-parameter design achieved 99.93% accuracy while already fixing six second-LayerNorm gains that are absorbed by the downstream MLP; this tests the analogous unused multiplicative gauge before removing additional empirically sensitive biases.

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
    """LayerNorm with gain and shift coordinate 3 absorbed by attention."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 1))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one = self.weight_rest.new_ones(1)
        zero = self.bias_rest.new_zeros(1)
        weight = torch.cat((self.weight_rest[:3], one, self.weight_rest[3:]))
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE