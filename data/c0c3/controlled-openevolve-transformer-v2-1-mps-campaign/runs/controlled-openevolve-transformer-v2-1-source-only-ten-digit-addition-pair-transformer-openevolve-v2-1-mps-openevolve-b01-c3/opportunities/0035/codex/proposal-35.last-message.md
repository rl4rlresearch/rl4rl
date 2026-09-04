MECHANISM: Five disjoint adaptive MLP bias ties

HYPOTHESIS: Sharing biases within five disjoint pairs of permutation-symmetric MLP neurons will produce a 1,408-parameter model with at least 99% accuracy, because the verified four-pair design achieved 99.98% while preserving all twelve neurons, independent weights, and learned thresholds.

INTENDED_EDIT: Parameterize the twelve MLP hidden biases with seven learned values, using the final five values for five disjoint neuron pairs.

EVIDENCE: The 1,409-parameter design with four disjoint adaptive bias ties achieved 99.98% accuracy; extending the same successful reduction by one pair is the smallest informative next ablation.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = MeanZeroLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
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


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = PairwiseTiedBiasLinear(d_model, d_ff)
        self.fc2 = MeanZeroLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE