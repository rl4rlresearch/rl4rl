MECHANISM: Three disjoint adaptive MLP bias ties

HYPOTHESIS: Sharing biases within three disjoint pairs of permutation-symmetric MLP neurons will produce a 1,410-parameter model with at least 99% accuracy, because the verified second disjoint tie preserved 99.96% accuracy while retaining all twelve neurons, independent weights, and learned thresholds.

INTENDED_EDIT: Parameterize the twelve MLP hidden biases with nine learned values, using the final three values for three disjoint neuron pairs.

EVIDENCE: The 1,411-parameter design with two disjoint learned bias ties achieved 99.96% accuracy, identical to the one-tie and untied verified designs; this directly motivates testing one additional incremental tie.

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose final four outputs form two learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_biases = self.free_bias[-2:].repeat_interleave(2)
        bias = torch.cat((self.free_bias[:-2], shared_biases))
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose final six outputs form three learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_biases = self.free_bias[-3:].repeat_interleave(2)
        bias = torch.cat((self.free_bias[:-3], shared_biases))
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE