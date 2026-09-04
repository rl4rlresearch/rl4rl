MECHANISM: Single adaptive final-LayerNorm scale

HYPOTHESIS: Extending the verified seven-way final-scale tie to all eight channels will reduce the best qualified model from 1,246 to 1,245 parameters while retaining at least 99% accuracy, because the shared scale remains learned and continues to condition both final normalization and pre-attention normalization.

INTENDED_EDIT: Reconstruct the final LayerNorm from one learned scalar shared across all eight channels, leaving the verified four-terminal attention kernel and training process unchanged.

EVIDENCE: Seven-way final-scale tying achieved 99.80% accuracy with 1,246 parameters, and every preceding extension also exceeded 99%; this motivates the final isolated scale tie while preserving the adaptive signal whose complete removal previously underperformed.

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with its last five channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 4))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:].expand(4)))
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with one adaptive scale shared by all channels."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)
>>>>>>> REPLACE