MECHANISM: Three four-neuron adaptive threshold clusters

HYPOTHESIS: Sharing the twelve MLP hidden biases across three learned four-neuron clusters will produce a 1,404-parameter model with at least 99% accuracy.

INTENDED_EDIT: Replace four independent biases and four pair-shared biases with three learned biases, each shared by four MLP neurons.

EVIDENCE: The verified 1,405-parameter design achieved 99.40% using two quartets and two pairs; merging its two remaining pairs into a third quartet is the smallest next reduction along the consistently successful bias-sharing sequence.

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose final eight outputs form four learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_biases = self.free_bias[-4:].repeat_interleave(2)
        bias = torch.cat((self.free_bias[:-4], shared_biases))
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