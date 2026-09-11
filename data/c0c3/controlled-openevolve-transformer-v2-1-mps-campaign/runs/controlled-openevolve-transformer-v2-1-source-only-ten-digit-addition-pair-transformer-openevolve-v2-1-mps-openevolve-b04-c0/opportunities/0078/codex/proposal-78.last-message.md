MECHANISM: Fifth attention-input gain anchoring

HYPOTHESIS: Fixing `ln1.weight[5]` alongside coordinates 0, 2, 4, and 6 will reduce the model from 1588 to 1587 parameters while retaining at least 99% accuracy, because its scale is absorbable by QKV column 5 and coordinate-specific outcomes leave untested coordinate 5 informative despite earlier failures at odd coordinates 1 and 3.

INTENDED_EDIT: Store three learned first-LayerNorm gains and reconstruct gain coordinates 0, 2, 4, 5, and 6 as fixed ones, while continuing to fix shift coordinate 3.

EVIDENCE: The current design achieved 99.90% with gain coordinates 0, 2, 4, and 6 fixed; prior results also showed strong coordinate sensitivity, including `fc2.bias[0]` failing at 2.79% while coordinate 1 achieved 99.93%, so the next untested gain coordinate cannot be inferred solely from failures at coordinates 1 and 3.

<<<<<<< SEARCH
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gain coordinates 0, 2, 4, and 6 and shift coordinate 3 fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 4))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        one = self.weight_rest.new_ones(1)
        weight = torch.cat(
            (
                one,
                self.weight_rest[:1],
                one,
                self.weight_rest[1:2],
                one,
                self.weight_rest[2:3],
                one,
                self.weight_rest[3:],
            )
        )
=======
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gains 0, 2, 4, 5, and 6 and shift 3 fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 5))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        one = self.weight_rest.new_ones(1)
        weight = torch.cat(
            (
                one,
                self.weight_rest[:1],
                one,
                self.weight_rest[1:2],
                one,
                one,
                one,
                self.weight_rest[2:],
            )
        )
>>>>>>> REPLACE