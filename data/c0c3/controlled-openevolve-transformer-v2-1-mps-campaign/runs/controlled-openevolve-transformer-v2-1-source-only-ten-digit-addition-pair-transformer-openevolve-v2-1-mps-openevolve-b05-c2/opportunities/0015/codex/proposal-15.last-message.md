MECHANISM: Incremental attention value/output-bias redundancy anchoring

HYPOTHESIS: Fixing a third mean-zero attention projection-bias coordinate while retaining all eight value-bias coordinates will produce a 1579-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reduce the attention output projection’s learned mean-zero bias from five coordinates to four, reconstructing the final three coordinates as zero.

EVIDENCE: Two projection-bias anchors achieved 99.97% accuracy with 1580 parameters, improving on the one-anchor model’s 99.79%; this supports testing one additional coordinate without disturbing the sensitive value-bias pathway.

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
class ThriceAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with three bias coordinates absorbed by value bias."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 3))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = TwiceAnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = ThriceAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE