MECHANISM: Bias-free MLP hidden expansion

HYPOTHESIS: The resulting 1,145-parameter transformer will retain at least 99% accuracy because the verified 1,146-parameter model reached 99.99%, and this removes only its zero-initialized scalar MLP bias while preserving the successful attention, normalization, projections, and positional routing.

INTENDED_EDIT: Remove the learned tilted hidden-quartet bias and make the MLP input projection bias-free, reducing the model by one parameter.

EVIDENCE: The current 1,146-parameter design achieved 99.99%; prior reductions to relative biases, final normalization, and projection subspaces failed, so the remaining isolated MLP hidden-bias scalar is the most informative next target.

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with a tilted contrast between hidden-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 12:
            raise ValueError("out_features must be twelve")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_contrast = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_bias = torch.cat(
            (self.bias_contrast, (-self.bias_contrast / 3).expand(2))
        )
        bias = quartet_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Bias-free MLP hidden projection."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 12:
            raise ValueError("out_features must be twelve")
        super().__init__(in_features, out_features, bias=False)
>>>>>>> REPLACE