MECHANISM: Centered hidden-quartet contrast

HYPOTHESIS: A 1,270-parameter model will achieve at least 99% accuracy because it preserves the first-versus-final-quartets bias contrast available in the successful 1,271-parameter design while removing only the shared bias component retained by the failed global-sharing design.

INTENDED_EDIT: Replace the two learned MLP hidden-bias values with one learned centered contrast, assigning one value to the first quartet and its opposite half-value to each tied final quartet.

EVIDENCE: The 1,271-parameter design with biases of the form `[a, b, b]` achieved 99.04%, whereas the 1,270-parameter `[a, a, a]` design achieved only 96.61%; this directly motivates retaining the lost quartet contrast while constraining away the orthogonal common component.

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
    """Linear layer with a centered first-versus-final-quartets bias contrast."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 12:
            raise ValueError("out_features must be twelve")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_bias = torch.cat(
            (self.free_bias, -0.5 * self.free_bias.expand(2))
        )
        bias = quartet_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE