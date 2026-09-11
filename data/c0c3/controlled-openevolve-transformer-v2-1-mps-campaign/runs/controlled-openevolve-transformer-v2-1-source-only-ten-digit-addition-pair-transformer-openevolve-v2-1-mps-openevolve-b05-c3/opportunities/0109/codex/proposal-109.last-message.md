MECHANISM: Two-way final-LayerNorm bias tying

HYPOTHESIS: Tying the final two bias coordinates will reduce the qualified 1,131-parameter model to 1,130 parameters while retaining at least 99% accuracy, because the adaptive normalization scale is preserved and only one latent output-bias degree of freedom is removed.

INTENDED_EDIT: Reconstruct the zero-sum final LayerNorm bias from six learned coordinates with its final two coordinates tied, leaving attention, training, gauge-aware optimization, checkpoints, and decoding unchanged.

EVIDENCE: The current quotient design achieved 99.81% accuracy with 1,131 parameters, while sourcing its adaptive scale elsewhere collapsed to 88.17%; this motivates preserving the scale and testing an isolated final-bias tie instead.

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and zero-sum quotient bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(1)))
        return anchored - anchored.mean()
=======
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
>>>>>>> REPLACE