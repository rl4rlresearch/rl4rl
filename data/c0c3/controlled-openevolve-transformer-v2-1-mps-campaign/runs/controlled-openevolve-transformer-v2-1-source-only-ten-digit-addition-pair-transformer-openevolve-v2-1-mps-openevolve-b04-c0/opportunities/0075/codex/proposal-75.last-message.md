MECHANISM: Alternative second attention-input gain anchoring

HYPOTHESIS: Fixing `ln1.weight[2]` alongside coordinate 0 will reduce the model from 1591 to 1590 parameters while retaining at least 99% accuracy, because LayerNorm-gain removability is coordinate-sensitive and the alternative coordinate 1 narrowly missed at 98.89%.

INTENDED_EDIT: Store six learned first-LayerNorm gains and reconstruct gain coordinates 0 and 2 as fixed ones, while continuing to fix shift coordinate 3.

EVIDENCE: Fixing gain coordinate 0 achieved 99.91% at 1591 parameters, while additionally fixing coordinate 1 reached 98.89%; prior `fc2.bias` results showed that changing the anchored coordinate can move accuracy from 2.79% to 99.93%, motivating the next untested gain coordinate.

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
    """LayerNorm with gain coordinates 0 and 2 and shift coordinate 3 fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 2))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        one = self.weight_rest.new_ones(1)
        weight = torch.cat((one, self.weight_rest[:1], one, self.weight_rest[1:]))
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE