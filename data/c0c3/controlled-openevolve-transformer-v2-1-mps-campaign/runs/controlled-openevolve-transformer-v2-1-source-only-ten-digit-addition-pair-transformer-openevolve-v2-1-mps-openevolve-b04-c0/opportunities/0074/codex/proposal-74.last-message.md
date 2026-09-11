MECHANISM: Second attention-input gain anchoring

HYPOTHESIS: Fixing `ln1.weight[1]` at one will reduce the model from 1591 to 1590 parameters while retaining at least 99% accuracy, because fixing coordinate 0 achieved 99.91% and each LayerNorm gain can be absorbed into the corresponding learned QKV input column.

INTENDED_EDIT: Store six learned first-LayerNorm gains and reconstruct gain coordinates 0 and 1 as fixed ones, while continuing to fix shift coordinate 3.

EVIDENCE: The current 1591-parameter design achieved 99.91% after fixing `ln1.weight[0]`; the earlier failure at coordinate 3 demonstrates coordinate-sensitive optimization, making adjacent untested coordinate 1 the most informative next reduction.

<<<<<<< SEARCH
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
=======
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gains 0 and 1 and shift coordinate 3 fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 2))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        ones = self.weight_rest.new_ones(2)
        weight = torch.cat((ones, self.weight_rest))
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE