MECHANISM: Five-coordinate pre-MLP LayerNorm-scale quotient

HYPOTHESIS: Extending the verified four-coordinate `ln2` scale quotient to five coordinates will reduce the model from 1,282 to 1,281 learned parameters while retaining at least 99% accuracy, because `fc1` can absorb the additional fixed channel scale.

INTENDED_EDIT: Represent `ln2` with three learned scales and five fixed unit scales, leaving the learned attention, initialization, optimizer, checkpoints, and decoding interface unchanged.

EVIDENCE: The four-coordinate `ln2` quotient achieved 99.97% accuracy at 1,282 parameters after every one-through-three-coordinate variant exceeded 99%; the next one-coordinate extension is the most direct boundary test.

<<<<<<< SEARCH
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with four scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 4))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(4)))
=======
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with five scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 5))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(5)))
>>>>>>> REPLACE