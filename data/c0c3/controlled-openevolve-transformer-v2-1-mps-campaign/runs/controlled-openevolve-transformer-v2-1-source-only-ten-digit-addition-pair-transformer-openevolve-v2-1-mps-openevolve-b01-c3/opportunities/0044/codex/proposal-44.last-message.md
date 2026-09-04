MECHANISM: Adaptive final-normalization gain tie

HYPOTHESIS: Tying two learned final LayerNorm gains in the verified three-quartet MLP design will produce a 1,403-parameter model with at least 99% accuracy, because the shared gain remains adaptive and the initialized model function is unchanged.

INTENDED_EDIT: Use three learned four-neuron MLP bias groups and replace two independent final LayerNorm gains with one shared learned gain.

EVIDENCE: The 1,404-parameter three-quartet design achieved 99.57%. Prior 1,403 failures constrained MLP thresholds or projection weights; this tests a distinct one-scalar reduction while retaining adaptivity, which the successful MLP bias-sharing sequence indicates is preferable to fixing a parameter.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with all but one scale learned."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with one fixed scale and one learned scale pair."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_weight = self.weight[-1:].expand(2)
        weight = torch.cat(
            (self.weight[:-1], shared_weight, self.weight.new_ones(1))
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
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