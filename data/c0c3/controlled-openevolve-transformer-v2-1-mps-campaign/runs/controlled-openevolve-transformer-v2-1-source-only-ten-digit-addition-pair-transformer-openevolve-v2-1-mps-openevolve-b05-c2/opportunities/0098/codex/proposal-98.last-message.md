MECHANISM: Fourth MLP output-bias coordinate anchoring

HYPOTHESIS: Extending the qualified 1498-parameter design by fixing reduced MLP output-bias coordinate 3 will yield 1497 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reconstruct MLP output-bias coordinates 0–3 as zero while learning the remaining three reduced coordinates.

EVIDENCE: Reference Design 1 achieved 99.94% accuracy at 1498 parameters after successive anchoring of reduced MLP output-bias coordinates 0, 1, and 2, motivating coordinate 3 as the closest supported next reduction.

<<<<<<< SEARCH
class FirstAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinate 0 fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (1, 0))
        return F.linear(x, self.basis @ self.weight, bias)
=======
class FirstFourAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0 through 3 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[4:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (4, 0))
        return F.linear(x, self.basis @ self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = FirstAnchoredMeanZeroOutputLinear(d_ff, d_model)
=======
        self.fc2 = FirstFourAnchoredMeanZeroOutputLinear(d_ff, d_model)
>>>>>>> REPLACE