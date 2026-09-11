MECHANISM: Seven-way final-LayerNorm bias tying

HYPOTHESIS: Extending the qualified six-way final-bias tie to seven channels will reduce the model from 1,126 to 1,125 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.

INTENDED_EDIT: Reconstruct the zero-sum final LayerNorm bias from one learned coordinate with its final seven coordinates tied.

EVIDENCE: Six-way final-bias tying achieved 99.75% accuracy with 1,126 parameters after two-way through five-way ties all exceeded 99%, motivating one further isolated tie despite the reduced accuracy margin.

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and two-way-tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(2)))
        return anchored - anchored.mean()
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and seven-way-tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(7)))
        return anchored - anchored.mean()
>>>>>>> REPLACE