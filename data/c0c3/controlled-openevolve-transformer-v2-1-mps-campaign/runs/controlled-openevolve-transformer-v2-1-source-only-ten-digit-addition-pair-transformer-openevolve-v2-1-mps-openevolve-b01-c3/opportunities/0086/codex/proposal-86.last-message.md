MECHANISM: Independent MLP output-bias pair tying

HYPOTHESIS: A 1,269-parameter model will achieve at least 99% accuracy because it retains the verified 1,270-parameter tilted hidden-bias design while applying the already successful pair-tied mean-zero bias parameterization independently to the MLP output.

INTENDED_EDIT: Replace the three MLP hidden-quartet biases with the verified one-scalar tilted contrast, then tie one pair of MLP output-bias coordinates, reducing the current model by three parameters.

EVIDENCE: The tilted hidden-quartet contrast achieved 99.31% at 1,270 parameters, and every qualified design already uses pair tying successfully in the attention output projection; unlike failed 1,269-parameter experiments, this neither couples modules nor changes positional biases, normalization, or dense weights.

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

<<<<<<< SEARCH
        self.fc2 = MeanZeroLinear(d_ff, d_model)
=======
        self.fc2 = PairTiedBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE