MECHANISM: Third MLP output-bias coordinate anchoring

HYPOTHESIS: Extending the qualified 1499-parameter design by fixing reduced MLP output-bias coordinate 2 will yield 1498 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reconstruct MLP output-bias coordinates 0, 1, and 2 as zero while learning the remaining four reduced coordinates.

EVIDENCE: Successive anchoring of MLP output-bias coordinates 0 and 1 retained 99.87% and 99.84% accuracy respectively; coordinate 2 is the closest untested continuation of this successful reduction.

<<<<<<< SEARCH
class FirstTwoAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0 and 1 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (2, 0))
        return F.linear(x, self.basis @ self.weight, bias)
=======
class FirstThreeAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0, 1, and 2 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[3:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (3, 0))
        return F.linear(x, self.basis @ self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = FirstTwoAnchoredMeanZeroOutputLinear(d_ff, d_model)
=======
        self.fc2 = FirstThreeAnchoredMeanZeroOutputLinear(d_ff, d_model)
>>>>>>> REPLACE