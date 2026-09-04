MECHANISM: Quintet-tied attention output bias

HYPOTHESIS: A 1,060-parameter transformer will maintain at least 99% accuracy because the verified 1,061-parameter model achieved 99.98%, and this adds only one equality constraint to its quartet-tied attention output bias while preserving the load-bearing scalar MLP bias, feature-wise head adapter, learned matrices, and positional routing.

INTENDED_EDIT: Adopt the verified mean-zero MLP input projection and scalar MLP output bias, then reduce the learned attention output bias from four scalars to three by expanding its final scalar across five of the seven mean-zero coordinates.

EVIDENCE: The 1,062-parameter triplet-tied attention-bias model reached 99.94%, and extending that tie to a quartet produced 99.98% at 1,061 parameters; in contrast, removing the scalar MLP bias or collapsing the feature-wise head adapter failed, making one further attention-bias tie the most supported isolated reduction.

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
        self.output_bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(2)))
=======
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(5)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = PairTiedBiasMeanZeroLinear(d_ff, d_model)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = ScalarBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE