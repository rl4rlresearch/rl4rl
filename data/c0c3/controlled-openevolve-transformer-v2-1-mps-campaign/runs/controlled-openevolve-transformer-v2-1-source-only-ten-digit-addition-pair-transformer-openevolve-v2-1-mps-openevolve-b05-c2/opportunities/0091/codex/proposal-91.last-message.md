MECHANISM: Nonadjacent attention-output bias anchoring

HYPOTHESIS: Fixing the first reduced attention-output bias coordinate while retaining the two qualified final anchors will yield 1503 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Replace the twice-anchored attention projection with a three-anchor variant that reconstructs reduced bias coordinates 0, 5, and 6 as zero while learning coordinates 1–4.

EVIDENCE: Fixing reduced coordinates 5 and 6 achieved 99.97% accuracy at 1504 parameters, whereas additionally fixing adjacent coordinate 4 fell to 39.49%; testing coordinate 0 distinguishes coordinate sensitivity from a general 1503-parameter capacity limit.

<<<<<<< SEARCH
class TwiceAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with two bias coordinates absorbed by value bias."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 2))
        return F.linear(x, weight, bias)
=======
class SplitThreeAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with three noncontiguous bias coordinates fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[1:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (1, 2))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = MeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = SplitThreeAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE