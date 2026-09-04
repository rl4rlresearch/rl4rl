MECHANISM: Six-way final-LayerNorm bias tying

HYPOTHESIS: Extending the qualified five-way final-bias tie to six channels will reduce the model from 1,127 to 1,126 learned parameters while retaining at least 99% accuracy, because the adaptive normalization scale and all attention, value, MLP, and token-representation capacity remain unchanged.

INTENDED_EDIT: Reconstruct the zero-sum final LayerNorm bias from two learned coordinates with its final six coordinates tied.

EVIDENCE: Five-way final-bias tying achieved 99.90% accuracy with 1,127 parameters, following successful two-way through four-way ties; this motivates one further isolated bias tie while preserving components whose sharing or scale relocation caused severe accuracy collapse.

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and three-way-tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 3))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(3)))
        return anchored - anchored.mean()
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and six-way-tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 6))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(6)))
        return anchored - anchored.mean()
>>>>>>> REPLACE