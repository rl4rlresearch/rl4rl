MECHANISM: Quadratically coupled hidden-quartet bias

HYPOTHESIS: A 1,270-parameter model will achieve at least 99% accuracy because it retains the successful `[a, b, b]` hidden-bias structure and augments the narrowly failing centered contrast with a learned-magnitude-dependent common component.

INTENDED_EDIT: Replace three independent MLP quartet biases with one scalar producing a centered contrast plus a quadratic common shift, reducing the current model by two parameters.

EVIDENCE: The two-parameter `[a, b, b]` design achieved 99.04%, while the one-parameter pure centered contrast reached 98.94%; the small shortfall motivates restoring common-mode variation without adding another learned parameter.

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose outputs form learned-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with quadratically coupled learned-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 12:
            raise ValueError("out_features must be twelve")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contrast = self.free_bias
        common = contrast.square()
        quartet_bias = torch.cat(
            (common + contrast, (common - 0.5 * contrast).expand(2))
        )
        bias = quartet_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE