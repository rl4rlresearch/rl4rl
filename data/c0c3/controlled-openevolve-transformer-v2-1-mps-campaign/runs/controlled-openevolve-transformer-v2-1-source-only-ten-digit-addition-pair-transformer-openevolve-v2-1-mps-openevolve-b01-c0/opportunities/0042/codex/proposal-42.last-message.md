MECHANISM: Pre-MLP LayerNorm bias/input-bias redundancy

HYPOTHESIS: Removing a third `ln2` bias degree of freedom will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because its zero-initialized additive effect is representable by the downstream `fc1` bias without the multiplicative optimization disruption seen when fixing an `ln2` scale.

INTENDED_EDIT: Store five learned `ln2` bias contrasts and reconstruct two anchored coordinates before centering the seven active coordinates; leave `ln1` unchanged.

EVIDENCE: Balanced LayerNorm common-bias gauges previously passed, whereas fixing the eighth `ln2` scale reached only 73.64%; this extends the successful additive-bias mechanism while preserving the initial function and RNG sequence.

<<<<<<< SEARCH
class ReducedBiasLayerNorm(nn.Module):
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
=======
class ReducedBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 3))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(2)))
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE