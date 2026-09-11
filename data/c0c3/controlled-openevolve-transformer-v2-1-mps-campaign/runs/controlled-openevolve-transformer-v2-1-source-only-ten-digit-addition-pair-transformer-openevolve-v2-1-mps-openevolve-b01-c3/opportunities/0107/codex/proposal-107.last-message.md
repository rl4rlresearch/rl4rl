MECHANISM: Scalar-shared MLP output bias

HYPOTHESIS: A 1,063-parameter transformer will maintain at least 99% accuracy because the verified 1,064-parameter model achieved 100%, and this adds only one equality constraint to its triplet-and-quartet MLP output bias while preserving all learned matrices, attention routing, normalization, and residual paths.

INTENDED_EDIT: Adopt the verified mean-zero MLP input factorization and replace the two-scalar triplet-and-quartet output bias with one scalar shared across all seven basis coordinates.

EVIDENCE: The 1,064-parameter triplet-and-quartet bias design achieved 100% after successive MLP output-bias sharing reductions all retained at least 99% accuracy, motivating one final isolated bias-sharing reduction.

<<<<<<< SEARCH
class TwoPairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with two adaptively shared bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 5:
            raise ValueError("out_features must be at least five")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.free_bias[:-2], self.free_bias[-2:].repeat_interleave(2))
        )
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
=======
class ScalarBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with one scalar shared across bias coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.expand(self.linear.out_features)
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = TwoPairTiedBiasMeanZeroLinear(d_ff, d_model)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = ScalarBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE