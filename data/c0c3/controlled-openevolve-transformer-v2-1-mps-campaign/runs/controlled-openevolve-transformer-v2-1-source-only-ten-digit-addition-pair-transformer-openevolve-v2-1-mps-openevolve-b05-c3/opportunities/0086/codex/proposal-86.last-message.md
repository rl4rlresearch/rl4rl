MECHANISM: Six-coordinate pre-MLP LayerNorm-scale quotient

HYPOTHESIS: Extending the verified five-coordinate `ln2` scale quotient to six coordinates will reduce the model from 1,281 to 1,280 learned parameters while retaining at least 99% accuracy, because `fc1` can absorb the additional fixed channel scale while two adaptive scales remain.

INTENDED_EDIT: Represent `ln2` with two learned scales and six fixed unit scales, leaving the learned attention, optimizer, checkpoints, and decoding interface unchanged.

EVIDENCE: The five-coordinate `ln2` quotient achieved 99.94% accuracy with 1,281 parameters after every one-through-four-coordinate variant also exceeded 99%; the next single-coordinate extension is the most direct boundary test.

<<<<<<< SEARCH
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with three scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 3))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(3)))
=======
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with six scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 6))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(6)))
>>>>>>> REPLACE