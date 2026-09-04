MECHANISM: Second attention-output bias anchoring

HYPOTHESIS: Fixing a second reduced attention-output bias coordinate will produce a 1504-parameter model with at least 99% accuracy.

INTENDED_EDIT: Retain five of seven reduced attention-output bias coordinates, reconstruct the final two as zero, and use this twice-anchored projection in causal self-attention.

EVIDENCE: Reference Design 1 achieved 99.97% accuracy with 1504 parameters using this exact reduction, while fixing a third coordinate fell to 39.49%.

<<<<<<< SEARCH
class AnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with one reduced bias coordinate fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 1))
        return F.linear(x, weight, bias)
=======
class TwiceAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with two reduced bias coordinates fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 2))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = AnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = TwiceAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE