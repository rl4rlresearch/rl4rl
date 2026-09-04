MECHANISM: Adaptive attention-output bias tie

HYPOTHESIS: Tying two learned attention-output bias coordinates will produce a 1,403-parameter model with at least 99% accuracy while preserving all projection weights and an adaptive bias in every output direction.

INTENDED_EDIT: Replace the attention output projection with a mean-zero linear layer whose final two internal bias coordinates share one learned scalar; leave the MLP and final normalization unchanged.

EVIDENCE: The 1,404-parameter three-quartet design achieved 99.57%; prior 1,403 failures altered MLP thresholds, projection weights, or final-normalization gains, motivating an isolated one-scalar test in the previously untested attention-output bias.

<<<<<<< SEARCH
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


class AnchoredScaleLayerNorm(nn.Module):
=======
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


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


class AnchoredScaleLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = MeanZeroLinear(d_model, d_model)
=======
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE