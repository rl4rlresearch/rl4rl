MECHANISM: Five-way final-LayerNorm bias tying

HYPOTHESIS: Extending the qualified four-way final-bias tie to five channels will reduce the model from 1,128 to 1,127 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.

INTENDED_EDIT: Reconstruct the zero-sum final LayerNorm bias from three learned coordinates with its final five coordinates tied.

EVIDENCE: Four-way final-bias tying achieved 99.83% accuracy with 1,128 parameters, following successful two-way and three-way ties at 99.90% and 99.83%; this motivates one further isolated bias tie while preserving components whose sharing or scale relocation caused severe accuracy collapse.

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and four-way-tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 4))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(4)))
        return anchored - anchored.mean()
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and five-way-tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 5))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(5)))
        return anchored - anchored.mean()
>>>>>>> REPLACE