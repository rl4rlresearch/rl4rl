MECHANISM: Dual four-neuron adaptive threshold clusters

HYPOTHESIS: Sharing MLP biases in two four-neuron clusters and two pairs will produce a 1,405-parameter model with at least 99% accuracy, because the verified 1,406-parameter design achieved 99.87% and this imposes only one additional scalar tie while preserving every neuron, weight, and learned threshold.

INTENDED_EDIT: Replace five learned MLP bias groups—one quartet and four pairs—with four learned groups comprising two quartets and two pairs.

EVIDENCE: The 1,406-parameter design with one four-neuron bias cluster and four bias pairs achieved 99.87%; merging two remaining pairs into a second quartet is the smallest incremental reduction along the demonstrated successful bias-sharing sequence.

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose final ten outputs form five learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_biases = self.free_bias[-5:].repeat_interleave(2)
        bias = torch.cat((self.free_bias[:-5], shared_biases))
        return F.linear(x, self.weight, bias)
=======
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
>>>>>>> REPLACE