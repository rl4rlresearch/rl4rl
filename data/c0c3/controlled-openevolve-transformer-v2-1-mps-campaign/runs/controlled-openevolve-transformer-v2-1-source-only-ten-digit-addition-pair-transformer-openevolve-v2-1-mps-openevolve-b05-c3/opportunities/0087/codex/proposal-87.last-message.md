MECHANISM: Seven-coordinate pre-MLP LayerNorm-scale quotient

HYPOTHESIS: Extending the verified six-coordinate `ln2` scale quotient to seven coordinates will reduce the model from 1,280 to 1,279 learned parameters while retaining at least 99% accuracy, because `fc1` can absorb the additional fixed channel scale while one adaptive scale remains.

INTENDED_EDIT: Represent `ln2` with one learned scale and seven fixed unit scales, preserving the learned attention, gauge-aware optimizer, checkpoints, and decoding interface.

EVIDENCE: The six-coordinate `ln2` quotient achieved 99.87% accuracy with 1,280 parameters, and the analogous seven-coordinate `ln1` quotient achieved 99.96%; this directly motivates testing the next single-coordinate reduction without removing all adaptive scales.

<<<<<<< SEARCH
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with two scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 2))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(2)))
=======
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with seven scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(7)))
>>>>>>> REPLACE