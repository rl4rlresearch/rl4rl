MECHANISM: Linearly tilted hidden-quartet contrast

HYPOTHESIS: A 1,270-parameter model will achieve at least 99% accuracy because adding a small linear common-mode component to the narrowly failing 98.94% centered contrast preserves its useful quartet distinction while avoiding the quadratic coupling’s zero initial common-mode gradient.

INTENDED_EDIT: Replace the two learned MLP hidden-bias values with one scalar that assigns the first quartet its full value and each tied final quartet negative one-third of that value.

EVIDENCE: The one-parameter centered `[a, -a/2, -a/2]` design reached 98.94%, much closer to the threshold than global sharing at 96.61%; the quadratic common shift fell to 91.42%, motivating a small linear tilt that is trainable from initialization.

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with bias quartets, sharing the final two quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 8 or out_features % 4 != 0:
            raise ValueError("out_features must be a multiple of four and at least eight")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4 - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_bias = torch.cat((self.free_bias, self.free_bias[-1:]))
        bias = quartet_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with a tilted contrast between hidden-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 12:
            raise ValueError("out_features must be twelve")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_contrast = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_bias = torch.cat(
            (self.bias_contrast, (-self.bias_contrast / 3).expand(2))
        )
        bias = quartet_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE