MECHANISM: Two disjoint adaptive attention-output bias ties

HYPOTHESIS: A 1,402-parameter model will retain at least 99% accuracy because the verified 1,403-parameter design achieved 99.98% with one adaptive attention-output bias pair, and adding a second disjoint pair preserves every projection weight and an adaptive bias in every internal output direction.

INTENDED_EDIT: Share the MLP hidden biases across three learned quartets and parameterize the attention output projection’s seven internal bias coordinates with three independent values and two learned pair-shared values.

EVIDENCE: The 1,403-parameter design combining three MLP-bias quartets with one attention-output bias tie achieved 99.98%; extending that successful, previously robust mechanism by one disjoint bias tie is the smallest informative next reduction.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
=======
class TwoPairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with two disjoint adaptive bias pairs."""

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


class AnchoredScaleLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = MeanZeroLinear(d_model, d_model)
=======
        self.proj = TwoPairTiedBiasMeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with two learned-bias quartets and learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 8 or out_features % 2 != 0:
            raise ValueError("out_features must be even and at least eight")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 2 - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_biases = self.free_bias[:2].repeat_interleave(4)
        paired_biases = self.free_bias[2:].repeat_interleave(2)
        bias = torch.cat((quartet_biases, paired_biases))
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