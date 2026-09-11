MECHANISM: Fourth even-coordinate attention-input gain anchoring

HYPOTHESIS: Fixing `ln1.weight[6]` alongside coordinates 0, 2, and 4 will reduce the model from 1589 to 1588 parameters while retaining at least 99% accuracy, because every tested even gain coordinate so far has been removable whereas coordinates 1 and 3 failed.

INTENDED_EDIT: Store four learned first-LayerNorm gains and reconstruct gain coordinates 0, 2, 4, and 6 as fixed ones, while continuing to fix shift coordinate 3.

EVIDENCE: Fixing gain coordinates 0, 2, and 4 achieved 99.88% at 1589 parameters; the successful even-coordinate sequence, contrasted with 98.89% for coordinate 1 and 74.01% for coordinate 3, makes coordinate 6 the most informative next reduction.

<<<<<<< SEARCH
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gain coordinates 0, 2, and 4 and shift coordinate 3 fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 3))
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
                self.weight_rest[2:],
            )
        )
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
=======
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
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE