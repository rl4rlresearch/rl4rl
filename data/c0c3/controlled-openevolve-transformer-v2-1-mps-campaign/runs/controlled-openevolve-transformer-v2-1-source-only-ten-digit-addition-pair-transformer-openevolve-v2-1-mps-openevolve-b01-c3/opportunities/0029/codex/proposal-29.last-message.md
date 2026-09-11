MECHANISM: Zero-anchored MLP hidden bias

HYPOTHESIS: Fixing one of twelve MLP hidden-unit biases at zero will reduce the model to 1,412 parameters while retaining at least 99% accuracy, because it preserves every hidden neuron and all learned weights while removing only one scalar threshold from a permutation-symmetric hidden layer.

INTENDED_EDIT: Replace the MLP’s first projection with an otherwise identical linear layer whose final output bias is fixed at zero and whose other eleven biases remain independently learned.

EVIDENCE: The 1,413-parameter design achieved 99.96% accuracy, whereas removing an entire MLP neuron and sixteen parameters still reached 98.2%; retaining that neuron’s fifteen weights while removing only its initially zero bias is a substantially more conservative capacity reduction. Unlike the failed terminal-scale tie, MLP hidden units have no feature-coordinate identity.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = MeanZeroLinear(d_ff, d_model)
=======
class AnchoredBiasLinear(nn.Linear):
    """Linear layer with its final output bias fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)
        self.free_bias = nn.Parameter(torch.zeros(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.free_bias, (0, 1))
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = AnchoredBiasLinear(d_model, d_ff)
        self.fc2 = MeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE