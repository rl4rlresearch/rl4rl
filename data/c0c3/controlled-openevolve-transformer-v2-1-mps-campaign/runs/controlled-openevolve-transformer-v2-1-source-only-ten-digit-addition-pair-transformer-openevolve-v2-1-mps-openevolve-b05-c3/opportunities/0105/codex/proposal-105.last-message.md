MECHANISM: Seven-way final-LayerNorm scale tying

HYPOTHESIS: Extending the verified six-way final-scale tie to seven channels will reduce the model from 1,247 to 1,246 learned parameters while retaining at least 99% accuracy, because two learned final-scale values and the shared adaptive pre-attention scale remain.

INTENDED_EDIT: Reconstruct the eight-channel final LayerNorm from two learned scales by sharing its last learned scale across the final seven channels.

EVIDENCE: Six-way final-scale tying achieved 99.83% accuracy with 1,247 parameters, and every preceding extension from two-way through five-way also exceeded 99%, motivating one further isolated scale tie while preserving the verified four-terminal attention kernel.

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with its last six channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:].expand(5)))
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with its last seven channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 6))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:].expand(6)))
>>>>>>> REPLACE