MECHANISM: Interior attention-output bias anchoring

HYPOTHESIS: Fixing reduced attention-output bias coordinate 3 in the qualified 1502-parameter design will yield 1501 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reconstruct reduced projection-bias coordinates 0, 1, 3, 5, and 6 as zero, leaving only coordinates 2 and 4 learned.

EVIDENCE: The 1502-parameter design anchoring coordinates 0, 1, 5, and 6 achieved 99.95% accuracy; coordinate 2 then failed at 74.27%, and coordinate 4 was previously harmful, making coordinate 3 the only untested extension of that qualified bias pattern.

<<<<<<< SEARCH
class SplitThreeAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with three noncontiguous bias coordinates fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[1:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (1, 2))
        return F.linear(x, weight, bias)
=======
class InteriorFiveAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with five split bias coordinates fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(
            torch.cat((full_bias[2:3], full_bias[4:5]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        middle_bias = torch.cat(
            (
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
            )
        )
        bias = self.basis @ F.pad(middle_bias, (2, 2))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = SplitThreeAnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = InteriorFiveAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE