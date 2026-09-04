MECHANISM: Final-coordinate attention-input gain anchoring

HYPOTHESIS: Fixing `ln1.weight[7]` alongside coordinates 0, 2, 4, 5, and 6 will reduce the model from 1587 to 1586 parameters while retaining at least 99% accuracy, because coordinate 7 is the only untested first-LayerNorm gain and its scale is absorbable by QKV input column 7.

INTENDED_EDIT: Store only gains 1 and 3 as learned parameters and reconstruct all other first-LayerNorm gains as fixed ones, while continuing to fix shift coordinate 3.

EVIDENCE: The current design achieved 99.63% after fixing gains 0, 2, 4, 5, and 6; although additional fixes at gains 1 and 3 failed, prior coordinate-sensitive results show those failures do not determine whether the sole remaining coordinate 7 is removable.

<<<<<<< SEARCH
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gains 0, 2, 4, 5, and 6 and shift 3 fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 5))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))
=======
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gains 0, 2, 4, 5, 6, and 7 and shift 3 fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 6))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
                one,
                one,
                one,
                self.weight_rest[2:],
=======
                one,
                one,
                one,
                one,
>>>>>>> REPLACE