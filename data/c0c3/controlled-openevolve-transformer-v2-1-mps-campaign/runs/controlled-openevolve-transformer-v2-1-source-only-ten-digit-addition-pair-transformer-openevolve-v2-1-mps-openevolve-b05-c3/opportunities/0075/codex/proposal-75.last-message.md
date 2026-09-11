MECHANISM: Seven-coordinate pre-attention LayerNorm-scale quotient

HYPOTHESIS: Extending the verified six-coordinate `ln1` scale quotient to seven coordinates will reduce the model from 1,314 to 1,313 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while one adaptive scale preserves optimization flexibility absent from the failed fully non-affine design.

INTENDED_EDIT: Fix the final seven `ln1` scale coordinates at one, leaving one learned scale and preserving the learned attention, initialization, optimizer, checkpoints, and decoding interface.

EVIDENCE: The six-coordinate quotient achieved 99.97% accuracy with 1,314 parameters, while fixing all eight scales failed at 42.51%; testing seven fixed coordinates is the direct incremental experiment that locates the boundary.

<<<<<<< SEARCH
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with two scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 2))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(2)))
=======
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with seven scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(7)))
>>>>>>> REPLACE