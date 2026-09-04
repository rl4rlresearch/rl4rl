MECHANISM: Distributed adaptive residual-output bias sharing

HYPOTHESIS: A 1,402-parameter model will retain at least 99% accuracy because it keeps the verified attention-output bias pair and places the additional adaptive tie in the distinct MLP output projection, avoiding the failed second attention-output tie.

INTENDED_EDIT: Use three learned MLP bias quartets and share one internal mean-zero output-bias pair in both the attention and MLP residual projections.

EVIDENCE: The 1,403-parameter design with one attention-output bias tie achieved 99.98%, while adding a second tie in that same projection fell to 90.13%; distributing the next tie to the previously untested MLP output tests whether the failure was attention-coordinate-specific.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
=======
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

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with one learned-bias quartet and learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 4 or out_features % 2 != 0:
            raise ValueError("out_features must be even and at least four")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 2 - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_bias = self.free_bias[:1].repeat_interleave(4)
        paired_biases = self.free_bias[1:].repeat_interleave(2)
        bias = torch.cat((quartet_bias, paired_biases))
        return F.linear(x, self.weight, bias)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = MeanZeroLinear(d_ff, d_model)
=======
        self.fc2 = PairTiedBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE