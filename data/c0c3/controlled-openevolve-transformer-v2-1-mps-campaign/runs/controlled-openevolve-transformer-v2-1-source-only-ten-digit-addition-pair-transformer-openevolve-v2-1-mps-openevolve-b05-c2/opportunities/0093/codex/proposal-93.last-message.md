MECHANISM: Low-side fifth attention-output bias anchoring

HYPOTHESIS: Fixing reduced attention-output bias coordinate 2 will reduce the qualified model to 1501 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reconstruct reduced projection-bias coordinates 0, 1, 2, 5, and 6 as zero, leaving coordinates 3 and 4 learned.

EVIDENCE: Anchoring coordinates 0, 1, 5, and 6 achieved 99.95% accuracy at 1502 parameters, while anchoring coordinate 4 with 5 and 6 failed at 39.49%; coordinate 2 is the closest untested extension on the successful low-coordinate side and preserves coordinate 4.

<<<<<<< SEARCH
class LowSplitFourAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with four split bias coordinates fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[2:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (2, 2))
        return F.linear(x, weight, bias)
=======
class LowSplitFiveAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with five split bias coordinates fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[3:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (3, 2))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = LowSplitFourAnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = LowSplitFiveAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE