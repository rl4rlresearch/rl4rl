MECHANISM: Triplet-and-quartet-tied MLP output bias

HYPOTHESIS: A 1,064-parameter transformer will maintain at least 99% accuracy because the verified 1,065-parameter model reached 99.96%, and this adds only one equality constraint to its three-scalar MLP output bias while preserving all learned matrices, attention routing, normalization, and residual paths.

INTENDED_EDIT: Adopt the verified triplet-and-pair bias sharing, then tie its two bias-pair scalars to form the seven-coordinate pattern `[a,a,a,b,b,b,b]`, reducing the model by one parameter.

EVIDENCE: The 1,065-parameter design achieved 99.96% after four successive MLP output-bias sharing reductions retained at least 99% accuracy, motivating one more isolated bias tie without altering previously sensitive components.

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
class TripletQuartetTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with a learned bias triplet and quartet."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 8:
            raise ValueError("out_features must be eight")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.free_bias[:1].expand(3), self.free_bias[1:].expand(4))
        )
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = TwoPairTiedBiasMeanZeroLinear(d_ff, d_model)
=======
        self.fc2 = TripletQuartetTiedBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE