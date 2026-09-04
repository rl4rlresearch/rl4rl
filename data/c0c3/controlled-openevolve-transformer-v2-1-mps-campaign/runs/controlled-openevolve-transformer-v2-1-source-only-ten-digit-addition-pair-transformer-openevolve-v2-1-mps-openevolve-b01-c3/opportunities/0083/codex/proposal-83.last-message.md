MECHANISM: Extended attention-output bias tying with tilted MLP contrast

HYPOTHESIS: A 1,269-parameter model will achieve at least 99% accuracy because it retains the verified 1,270-parameter tilted hidden-bias design while extending the already successful attention-output bias pair into a triplet.

INTENDED_EDIT: Use one scalar for the final three attention projection bias coordinates and replace the three MLP quartet biases with the verified tilted contrast, removing three parameters total.

EVIDENCE: The tilted-contrast design achieved 99.31% at 1,270 parameters, and all qualified designs already tolerate one tied attention-output bias pair; unlike the failed 1,269-parameter cross-module sharing experiment, this reduction keeps the MLP and attention parameters independent.

<<<<<<< SEARCH
class PairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with one adaptively shared bias pair."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 3:
            raise ValueError("out_features must be at least three")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.free_bias[:-1], self.free_bias[-1:].expand(2)))
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
=======
class PairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with one adaptively shared bias triplet."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 4:
            raise ValueError("out_features must be at least four")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.free_bias[:-1], self.free_bias[-1:].expand(3)))
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

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