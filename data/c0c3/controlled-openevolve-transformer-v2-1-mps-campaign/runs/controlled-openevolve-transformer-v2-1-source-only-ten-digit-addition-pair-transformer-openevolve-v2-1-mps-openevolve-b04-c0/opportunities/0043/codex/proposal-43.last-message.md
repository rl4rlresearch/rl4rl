MECHANISM: Second pre-attention shift-to-bias gauge fixing

HYPOTHESIS: Fixing pre-attention LayerNorm bias coordinate 4 at zero will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because its query and value shifts are absorbable by learned attention biases and its constant key shift cancels under softmax.

INTENDED_EDIT: Remove pre-attention bias coordinate 4 from the learned vector and reconstruct bias coordinates 3 and 4 as fixed zeros.

EVIDENCE: Fixing pre-attention bias coordinate 3 achieved 99.91% at 1607 parameters, whereas fixing its gain narrowly failed at 98.69%; coordinate 4 is the strongest next shift candidate because its pre-MLP bias fixing achieved 99.91% and its distributed embedding anchor achieved 99.94%.

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
    """LayerNorm with shift coordinates 3 and 4 absorbed by attention biases."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros(2)
        bias = torch.cat((self.bias_rest[:3], zeros, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE