MECHANISM: Learned octet-and-quartet MLP threshold sharing

HYPOTHESIS: Merging two of the three learned MLP bias quartets into one eight-neuron cluster will produce a 1,403-parameter model with at least 99% accuracy while preserving every hidden neuron and learned weight.

INTENDED_EDIT: Replace three quartet-shared MLP biases with two learned biases: one shared across eight neurons and one shared across four.

EVIDENCE: The verified 1,404-parameter design achieved 99.57% accuracy with three learned bias quartets; merging two existing groups removes one scalar along the consistently successful incremental bias-sharing sequence.

<<<<<<< SEARCH
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
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with one learned-bias octet and learned-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 8 or out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four and at least eight")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4 - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        octet_bias = self.free_bias[:1].repeat_interleave(8)
        quartet_biases = self.free_bias[1:].repeat_interleave(4)
        bias = torch.cat((octet_bias, quartet_biases))
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE