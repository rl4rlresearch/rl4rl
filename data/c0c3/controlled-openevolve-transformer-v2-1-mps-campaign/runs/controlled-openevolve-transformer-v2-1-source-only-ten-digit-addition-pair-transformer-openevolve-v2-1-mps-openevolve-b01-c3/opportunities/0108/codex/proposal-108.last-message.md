MECHANISM: Bias-free mean-zero MLP projections

HYPOTHESIS: The resulting 1,062-parameter transformer will maintain at least 99% accuracy because the verified 1,063-parameter model achieved 99.64%, and this removes only its final scalar-shared MLP output bias while preserving every learned matrix, attention route, normalization, and residual path.

INTENDED_EDIT: Restrict the MLP input projection to the verified seven-dimensional mean-zero basis and make its mean-zero output projection bias-free.

EVIDENCE: The 1,063-parameter scalar-shared-bias design reached 99.64%; successive reductions of the same MLP output bias retained at least 99% accuracy, making removal of its final scalar the most isolated next reduction.

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
class BiasFreeMeanZeroLinear(MeanZeroLinear):
    """Bias-free linear map whose outputs lie in the mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.linear.bias = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = PairTiedBiasMeanZeroLinear(d_ff, d_model)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = BiasFreeMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE