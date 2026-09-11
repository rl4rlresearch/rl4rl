MECHANISM: Zero-shift LayerNorm gain absorption into the QKV input column

HYPOTHESIS: Fixing attention LayerNorm gain coordinate 3 at one will produce a 1576-parameter model with at least 99% accuracy because its corresponding LayerNorm shift is already fixed at zero, making the gain exactly absorbable into QKV column 3 while leaving initialization unchanged.

INTENDED_EDIT: Remove the second learned attention LayerNorm gain and reconstruct coordinate 3 as one; retain the learned gain shared by coordinates 1 and 7.

EVIDENCE: The 1577-parameter model reached 99.13%, while attempted 1576-parameter changes involving query biases, fixed Q/K coefficients, reciprocal value scaling, and folded biases failed. This targets an untested exact downstream scale redundancy without reciprocal initialization or perturbing the verified initial function.

<<<<<<< SEARCH
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
=======
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gain 3 fixed and gain 7 sharing gain 1."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 7))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        one = self.weight_rest.new_ones(1)
        weight = torch.cat(
            (
                one,
                self.weight_rest[:1],
                one,
                one,
                one,
                one,
                one,
                self.weight_rest[:1].detach(),
            )
        )
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE