MECHANISM: Four-neuron adaptive threshold cluster

HYPOTHESIS: Merging two of the six learned MLP bias pairs into one four-neuron cluster will produce a 1,406-parameter model with at least 99% accuracy, because it preserves every neuron, weight, and adaptive threshold while imposing only one additional scalar tie.

INTENDED_EDIT: Replace six pairwise-shared MLP biases with five learned biases: one shared across four neurons and four shared across two neurons each.

EVIDENCE: The 1,407-parameter design achieved 99.08% after all six disjoint bias ties; merging two existing bias groups is the smallest incremental reduction along the only demonstrated successful parameter-reduction sequence.

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose outputs form learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 2 != 0:
            raise ValueError("out_features must be even")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.repeat_interleave(2)
        return F.linear(x, self.weight, bias)
=======
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
>>>>>>> REPLACE