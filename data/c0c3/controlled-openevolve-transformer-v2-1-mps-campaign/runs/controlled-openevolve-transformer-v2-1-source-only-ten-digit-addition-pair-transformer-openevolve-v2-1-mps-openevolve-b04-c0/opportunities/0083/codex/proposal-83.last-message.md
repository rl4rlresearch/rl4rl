MECHANISM: Gradient-isolated sensitive-gain reuse

HYPOTHESIS: Reusing learned gain 3 for gains 1 and 7 through detached views will reduce the model to 1585 parameters while retaining at least 99% accuracy, because detaching gain 7 previously improved sharing from 98.67% to 99.90%, and gain 3 is the most optimization-sensitive remaining gain.

INTENDED_EDIT: Store only first-LayerNorm gain 3, use it normally at coordinate 3, and reuse detached copies at coordinates 1 and 7 while keeping the other gains fixed.

EVIDENCE: Fixing gain 3 reached only 74.01%, so it should remain directly learned; meanwhile, gradient-isolating the gain-7 reuse eliminated the harmful coupling that caused ordinary sharing with gain 1 to miss the threshold.

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
    """LayerNorm with gains 1 and 7 sharing gradient-isolated gain 3."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 7))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        one = self.weight_rest.new_ones(1)
        shared = self.weight_rest.detach()
        weight = torch.cat(
            (
                one,
                shared,
                one,
                self.weight_rest,
                one,
                one,
                one,
                shared,
            )
        )
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE