MECHANISM: Balanced second attention LayerNorm-bias gauge

HYPOTHESIS: Constraining `ln1` bias to six centered contrasts over seven active coordinates will reduce the model to 1632 parameters while retaining at least 99% accuracy, because the current centered `ln1` bias achieved 99.94% and the identical two-constraint structure in `ln2` achieved 99.96%.

INTENDED_EDIT: Store six learned `ln1` bias contrasts, center them across seven active coordinates, and keep the eighth coordinate fixed at zero.

EVIDENCE: The 1633-parameter centered `ln1` bias design reached 99.94%; moreover, the balanced six-contrast `ln2` parameterization succeeded at 99.96% where directly fixing a second coordinate failed, motivating the same distributed reduction for `ln1`.

<<<<<<< SEARCH
class CenteredBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        bias = bias - bias.mean()
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class CenteredBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE