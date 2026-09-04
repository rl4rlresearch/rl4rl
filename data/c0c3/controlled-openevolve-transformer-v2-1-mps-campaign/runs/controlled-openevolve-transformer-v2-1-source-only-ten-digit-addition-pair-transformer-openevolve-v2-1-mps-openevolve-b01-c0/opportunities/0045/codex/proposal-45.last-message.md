MECHANISM: Fourth pre-attention LayerNorm bias/projection-bias redundancy

HYPOTHESIS: Removing a fourth `ln1` bias degree of freedom will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy, because its additive effect remains absorbable by the query and value/output bias paths while remaining softmax-null for keys.

INTENDED_EDIT: Store four learned `ln1` bias contrasts instead of five and reconstruct three anchored coordinates before centering the seven active coordinates.

EVIDENCE: Removing the third `ln1` bias degree achieved 99.87% accuracy at 1617 parameters; extending that successful pre-attention mechanism by one coordinate is the smallest informative reduction, particularly because the analogous fourth `ln2` reduction failed and establishes that success is pathway-specific.

<<<<<<< SEARCH
class CenteredBiasLayerNorm(nn.Module):
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
=======
class CenteredBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 4))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(3)))
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE