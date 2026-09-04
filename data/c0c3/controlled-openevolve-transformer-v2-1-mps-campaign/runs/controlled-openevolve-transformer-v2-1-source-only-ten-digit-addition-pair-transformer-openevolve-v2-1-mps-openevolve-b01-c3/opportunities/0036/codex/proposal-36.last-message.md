MECHANISM: Complete pairwise sharing of adaptive MLP thresholds

HYPOTHESIS: Sharing biases across all six disjoint pairs of permutation-symmetric MLP neurons will produce a 1,407-parameter model with at least 99% accuracy, because the verified five-pair design achieved 99.96% while retaining all twelve neurons, independent weights, and learned thresholds.

INTENDED_EDIT: Apply the verified fixed 0.02 query bias and parameterize the twelve MLP hidden biases as six independently learned pairwise-shared values.

EVIDENCE: The 1,408-parameter five-pair design reached 99.96% accuracy; tying its remaining two independent neuron biases is the smallest incremental reduction supported by the successful one-through-five-pair sequence.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(n_head, self.head_dim - 1))
        self.proj = MeanZeroLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = MeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(self.q_bias, (0, 1)).reshape(d_model)
        q = q + q_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(
            self.q_bias.expand(self.n_head, 1), (0, self.head_dim - 1)
        ).reshape(d_model)
        q = q + q_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = MeanZeroLinear(d_ff, d_model)
=======
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


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = PairwiseTiedBiasLinear(d_model, d_ff)
        self.fc2 = MeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE