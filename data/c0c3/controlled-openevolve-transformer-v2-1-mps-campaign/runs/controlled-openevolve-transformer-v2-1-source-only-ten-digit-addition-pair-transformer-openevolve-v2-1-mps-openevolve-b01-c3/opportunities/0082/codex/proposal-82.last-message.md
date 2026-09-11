MECHANISM: Direct zero-bias parameter sharing

HYPOTHESIS: A 1,269-parameter model will achieve at least 99% accuracy because it preserves the verified 1,270-parameter model’s tilted hidden-quartet bias, while sharing its scalar with an existing zero-initialized MLP output-bias coordinate that receives direct gradients.

INTENDED_EDIT: Replace the three independent hidden-quartet biases with the verified tilted contrast and derive its scalar from the first MLP output-bias coordinate, eliminating three deduplicated parameters without changing the dense projections or positional biases.

EVIDENCE: The independent tilted contrast achieved 99.31% at 1,270 parameters. The prior 1,269-parameter gauge-packing attempt reached only 89.14% after altering carrier-row initialization and optimization geometry; sharing an explicit zero-initialized bias avoids those disruptions.

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


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = PairwiseTiedBiasLinear(d_model, d_ff)
        self.fc2 = MeanZeroLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with a tilted quartet bias shared with another model bias."""

    def __init__(
        self, in_features: int, out_features: int, bias_source: nn.Parameter
    ):
        if out_features != 12:
            raise ValueError("out_features must be twelve")
        super().__init__(in_features, out_features, bias=False)
        self.bias_source = bias_source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_contrast = self.bias_source[:1]
        quartet_bias = torch.cat(
            (bias_contrast, (-bias_contrast / 3).expand(2))
        )
        bias = quartet_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc2 = MeanZeroLinear(d_ff, d_model)
        self.fc1 = PairwiseTiedBiasLinear(
            d_model, d_ff, self.fc2.linear.bias
        )
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE