MECHANISM: Alternative third attention-input gain anchoring

HYPOTHESIS: Fixing `ln1.weight[4]` alongside coordinates 0 and 2 will reduce the model from 1590 to 1589 parameters while retaining at least 99% accuracy, because LayerNorm-gain removability is strongly coordinate-sensitive and coordinate 2 preserved 99.87% whereas coordinates 1 and 3 failed.

INTENDED_EDIT: Store five learned first-LayerNorm gains and reconstruct gain coordinates 0, 2, and 4 as fixed ones, while continuing to fix shift coordinate 3.

EVIDENCE: The current design achieved 99.87% with gain coordinates 0 and 2 fixed; the 98.89% result for coordinate 1 and 74.01% result for coordinate 3 make the next untested coordinate 4 the most informative capacity test.

<<<<<<< SEARCH
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gain coordinates 0 and 2 and shift coordinate 3 fixed."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 2))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        one = self.weight_rest.new_ones(1)
        weight = torch.cat((one, self.weight_rest[:1], one, self.weight_rest[1:]))
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
=======
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
>>>>>>> REPLACE