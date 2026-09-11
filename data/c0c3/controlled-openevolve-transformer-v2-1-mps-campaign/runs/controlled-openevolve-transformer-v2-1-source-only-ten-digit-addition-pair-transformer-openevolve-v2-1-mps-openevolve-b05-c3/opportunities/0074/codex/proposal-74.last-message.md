MECHANISM: Six-coordinate pre-attention LayerNorm-scale quotient

HYPOTHESIS: Extending the verified five-coordinate `ln1` scale quotient to six coordinates will reduce the model from 1,315 to 1,314 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while two adaptive scales remain.

INTENDED_EDIT: Fix the final six `ln1` scale coordinates at one, leaving the learned attention, initialization, optimizer, checkpoints, and decoding interface unchanged.

EVIDENCE: The five-coordinate quotient achieved 99.97% accuracy with 1,315 parameters after every one-through-four-coordinate quotient also exceeded 99%; this incremental extension most directly locates the boundary suggested by the failed all-eight-scale removal.

<<<<<<< SEARCH
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with three scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 3))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(3)))
=======
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with six scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 6))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(6)))
>>>>>>> REPLACE