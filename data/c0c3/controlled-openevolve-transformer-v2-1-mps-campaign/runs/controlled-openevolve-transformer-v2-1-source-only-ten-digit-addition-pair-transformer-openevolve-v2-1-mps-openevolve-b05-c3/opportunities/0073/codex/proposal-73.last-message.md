MECHANISM: Five-coordinate pre-attention LayerNorm-scale quotient

HYPOTHESIS: Extending the verified four-coordinate `ln1` scale quotient to five coordinates will reduce the model from 1,316 to 1,315 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while three adaptive scales remain.

INTENDED_EDIT: Fix the final five `ln1` scale coordinates at one instead of the final four, leaving the learned attention, initialization, optimizer, checkpoints, and decoding interface unchanged.

EVIDENCE: The four-coordinate quotient achieved 99.96% accuracy at 1,316 parameters, after the one-, two-, and three-coordinate variants all exceeded 99%; this next one-coordinate extension most directly probes the boundary indicated by the failed all-eight-scale removal.

<<<<<<< SEARCH
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with four scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 4))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(4)))
=======
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with five scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 5))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(5)))
>>>>>>> REPLACE