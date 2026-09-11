MECHANISM: Two-coordinate pre-attention LayerNorm-scale quotient

HYPOTHESIS: Extending the verified one-coordinate `ln1` scale quotient to two coordinates will reduce the model from 1,319 to 1,318 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb both fixed channel scales while six adaptive scales retain substantially more optimization flexibility than the failed fully non-affine design.

INTENDED_EDIT: Fix the final two `ln1` scale coordinates at one instead of only the final coordinate, leaving the learned attention, initialization, optimizer, checkpoints, and decoding interface unchanged.

EVIDENCE: The one-coordinate quotient achieved 99.52% accuracy at 1,319 parameters, whereas removing all eight scales collapsed to 42.51%; the next single-coordinate extension most directly tests where the optimization-sensitive boundary begins.

<<<<<<< SEARCH
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with one scale absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(1)))
=======
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with two scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 2))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(2)))
>>>>>>> REPLACE