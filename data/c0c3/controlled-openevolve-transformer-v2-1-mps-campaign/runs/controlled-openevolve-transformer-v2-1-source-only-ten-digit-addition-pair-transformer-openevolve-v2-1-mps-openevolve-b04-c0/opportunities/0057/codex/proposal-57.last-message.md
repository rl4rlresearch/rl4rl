MECHANISM: Attention preprojection gain gauge fixing

HYPOTHESIS: Fixing coordinate 3 of the pre-attention LayerNorm gain at one will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because its scale can be learned equivalently by coordinate 3 of every QKV projection row.

INTENDED_EDIT: Replace the eight learned pre-attention LayerNorm gains with seven learned coordinates, reconstructing gain coordinate 3 as a fixed one while preserving the existing initialization and anchored shift.

EVIDENCE: The verified 1599-parameter design already absorbs six pre-MLP LayerNorm gains into its downstream projection, while removing `fc1.bias[8]` failed at 77.57%; applying the analogous exact gain redundancy on the previously untouched attention side is therefore the most informative alternative.

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