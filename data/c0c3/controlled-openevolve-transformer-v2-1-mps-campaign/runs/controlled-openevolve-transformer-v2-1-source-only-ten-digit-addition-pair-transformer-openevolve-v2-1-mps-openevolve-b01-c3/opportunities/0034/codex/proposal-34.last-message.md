MECHANISM: Four disjoint adaptive MLP bias ties

HYPOTHESIS: Sharing biases within four disjoint pairs of permutation-symmetric MLP neurons will produce a 1,409-parameter model with at least 99% accuracy, because the verified three-pair design reached 99.52% while retaining all twelve neurons, independent weights, and learned thresholds.

INTENDED_EDIT: Parameterize the twelve MLP hidden biases with eight learned values, using the final four values for four disjoint neuron pairs.

EVIDENCE: The 1,410-parameter design with three disjoint adaptive bias ties achieved 99.52%; extending the same successful reduction by one pair is the smallest informative next ablation.

<<<<<<< SEARCH
class TiedBiasLinear(nn.Linear):
    """Linear layer whose final two outputs share one learned bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.free_bias, self.free_bias[-1:]))
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = TiedBiasLinear(d_model, d_ff)
=======
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


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = PairwiseTiedBiasLinear(d_model, d_ff)
>>>>>>> REPLACE