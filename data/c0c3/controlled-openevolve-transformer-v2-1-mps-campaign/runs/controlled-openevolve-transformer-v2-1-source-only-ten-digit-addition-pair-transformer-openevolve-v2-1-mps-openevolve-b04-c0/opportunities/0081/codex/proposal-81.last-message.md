MECHANISM: Sensitive-coordinate adaptive gain sharing

HYPOTHESIS: Sharing `ln1.weight[7]` with learned gain coordinate 3 will reduce the model from 1587 to 1586 parameters while maintaining at least 99% accuracy, because coordinate 3 is substantially more optimization-sensitive than coordinate 1 and may provide a better adaptive scale for coordinate 7.

INTENDED_EDIT: Store only first-LayerNorm gains 1 and 3, and reuse gain 3 for coordinate 7 instead of learning coordinate 7 independently.

EVIDENCE: Fixing gain 7 reached 96.28%, while sharing it with gain 1 improved accuracy to 98.67%, showing that adaptive sharing recovers substantial performance. Independently fixing gain 3 caused the largest tested degradation, to 74.01%, making its learned scale the most informative remaining sharing candidate.

<<<<<<< SEARCH
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
=======
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gain 7 shared with gain 3 and five other gains fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 6))
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
                self.weight_rest[1:2],
            )
        )
>>>>>>> REPLACE