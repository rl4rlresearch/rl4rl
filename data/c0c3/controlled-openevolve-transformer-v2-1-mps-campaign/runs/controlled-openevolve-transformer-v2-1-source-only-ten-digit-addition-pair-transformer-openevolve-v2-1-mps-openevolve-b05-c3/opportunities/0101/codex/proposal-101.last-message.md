MECHANISM: Three-way final-LayerNorm scale tying

HYPOTHESIS: Extending the qualified final-LayerNorm tie from two channels to three will reduce the model from 1,251 to 1,250 learned parameters while retaining at least 99% accuracy, because six independent final scales and the shared adaptive `ln1` signal remain.

INTENDED_EDIT: Reconstruct the eight-channel final LayerNorm from six learned scales by sharing its last learned scale across the final three channels.

EVIDENCE: The current two-way final-scale tie achieved 99.77% accuracy with 1,251 parameters, while complete `ln2` scale absorption also retained 99.94%; this motivates another isolated non-routing scale tie instead of the fifth lag-logit tie that collapsed to 91.63%.

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with its last two channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 1))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:]))
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with its last three channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 2))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:].expand(2)))
>>>>>>> REPLACE