MECHANISM: Quartet-tied attention output bias

HYPOTHESIS: A 1,061-parameter transformer will maintain at least 99% accuracy because the verified 1,062-parameter model achieved 99.94%, and this adds only one equality constraint to its triplet-tied attention output bias while preserving the load-bearing scalar MLP bias, feature-wise head adapter, learned matrices, and positional routing.

INTENDED_EDIT: Adopt the verified scalar-shared MLP output bias and reduce the attention output bias from six learned scalars to four by expanding its final scalar across four of the seven mean-zero basis coordinates.

EVIDENCE: The 1,062-parameter design retained 99.94% after extending the attention output-bias pair to a triplet, whereas removing the scalar MLP bias produced only 98.71% and replacing the feature-wise head adapter with a scalar produced 97.11%; therefore one further attention-bias tie is the most isolated supported reduction.

<<<<<<< SEARCH
class TripletTwoPairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with one bias triplet and two bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 8:
            raise ValueError("out_features must be eight")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.free_bias[:1].expand(3), self.free_bias[1:].repeat_interleave(2))
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
        self.output_bias = nn.Parameter(torch.zeros(d_model - 2))
=======
        self.output_bias = nn.Parameter(torch.zeros(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(2)))
=======
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(4)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = TripletTwoPairTiedBiasMeanZeroLinear(d_ff, d_model)
=======
        self.fc2 = ScalarBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE