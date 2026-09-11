MECHANISM: Triplet-and-pair-tied MLP output bias

HYPOTHESIS: The resulting 1,065-parameter transformer will maintain at least 99% accuracy because the verified 1,066-parameter model achieved 99.95%, and this adds only one equality constraint to its three-times-pair-tied MLP output bias while preserving all learned matrices, attention routing, normalization, and residual paths.

INTENDED_EDIT: Replace the seven-coordinate MLP output bias pattern `[a,b,b,c,c,d,d]` with `[a,a,a,b,b,c,c]`, reducing the learned bias from four scalars to three.

EVIDENCE: The current 1,066-parameter design reached 99.95% after a third successive MLP output-bias tie; the next isolated reduction extends that demonstrated bias-sharing mechanism by one parameter without altering any weight matrix or previously sensitive component.

<<<<<<< SEARCH
class ThreePairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with three adaptively shared bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 7:
            raise ValueError("out_features must be at least seven")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.free_bias[:-3], self.free_bias[-3:].repeat_interleave(2))
        )
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = ThreePairTiedBiasMeanZeroLinear(d_ff, d_model)
=======
        self.fc2 = TripletTwoPairTiedBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE