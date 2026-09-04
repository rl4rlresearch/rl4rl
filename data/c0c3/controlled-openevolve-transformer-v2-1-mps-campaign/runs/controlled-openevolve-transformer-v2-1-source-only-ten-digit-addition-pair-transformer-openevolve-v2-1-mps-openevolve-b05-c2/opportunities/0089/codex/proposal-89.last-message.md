MECHANISM: Third attention-output bias anchoring

HYPOTHESIS: Fixing a third zero-initialized reduced attention-output bias coordinate will reduce the qualified model to 1503 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Retain four of the seven reduced attention-output bias coordinates and reconstruct the other three as zero.

EVIDENCE: Successive first and second attention-output bias anchors achieved 99.95% at 1505 parameters and 99.97% at 1504 parameters, making the adjacent third anchor the closest supported reduction.

<<<<<<< SEARCH
class TwiceAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with two reduced bias coordinates fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (0, 2))
        return F.linear(x, self.basis @ self.weight, bias)
=======
class ThriceAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with three reduced bias coordinates fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (0, 3))
        return F.linear(x, self.basis @ self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = TwiceAnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = ThriceAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE