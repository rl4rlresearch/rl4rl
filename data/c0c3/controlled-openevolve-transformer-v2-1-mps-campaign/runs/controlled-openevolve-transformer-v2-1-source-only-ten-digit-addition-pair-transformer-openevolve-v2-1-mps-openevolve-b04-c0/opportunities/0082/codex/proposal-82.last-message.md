MECHANISM: Gradient-isolated adaptive gain sharing

HYPOTHESIS: Reusing gain 1 for gain 7 while detaching the coordinate-7 branch will achieve at least 99% accuracy with 1586 parameters; ordinary sharing reached 98.67% versus 96.28% when fixed, suggesting the learned value helps but its additional gradient may create harmful optimization interference.

INTENDED_EDIT: Store only gains 1 and 3, reconstruct gain 7 from a detached view of gain 1, and keep gains 0, 2, 4, 5, and 6 fixed.

EVIDENCE: Sharing gain 7 with gain 1 recovered accuracy from 96.28% to 98.67%, whereas sharing with gain 3 fell to 90%; the existing embedding anchors also use detachment specifically to preserve shared values without additional gradient coupling.

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
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
=======
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gain 7 sharing a gradient-isolated copy of gain 1."""

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
                self.weight_rest[:1].detach(),
            )
        )
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE