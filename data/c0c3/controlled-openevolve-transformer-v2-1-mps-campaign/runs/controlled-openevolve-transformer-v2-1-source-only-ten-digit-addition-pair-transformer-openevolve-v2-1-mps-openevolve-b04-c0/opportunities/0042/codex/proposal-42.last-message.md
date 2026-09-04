MECHANISM: Paired pre-attention scale gauge fixing

HYPOTHESIS: Fixing pre-attention LayerNorm gain coordinate 3 to one will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because its bias is already zero and its scale is exactly absorbable by column 3 of the learned QKV projection.

INTENDED_EDIT: Remove pre-attention gain coordinate 3 from the learned vector and reconstruct it as a fixed one during the forward pass.

EVIDENCE: Fixing pre-attention bias coordinate 3 retained 99.91% accuracy at 1607 parameters, while fixing the corresponding pre-MLP gain retained 99.93%; this tests the same demonstrated scale-to-weight redundancy on the attention side without constraining a new coordinate.

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
    """LayerNorm with coordinate-3 scale and shift absorbed by attention."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 1))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        one = self.weight_rest.new_ones(1)
        weight = torch.cat((self.weight_rest[:3], one, self.weight_rest[3:]))
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE