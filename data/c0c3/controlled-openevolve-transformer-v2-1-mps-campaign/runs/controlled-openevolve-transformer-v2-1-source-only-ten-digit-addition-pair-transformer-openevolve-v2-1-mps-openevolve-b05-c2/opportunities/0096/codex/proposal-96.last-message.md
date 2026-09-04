MECHANISM: Second MLP output-bias coordinate anchoring

HYPOTHESIS: Extending the qualified 1500-parameter design by fixing reduced MLP output-bias coordinate 1 will yield 1499 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified attention projection with only reduced bias coordinates 2 and 4 learned, then reconstruct MLP output-bias coordinates 0 and 1 as zero.

EVIDENCE: Reference Design 1 achieved 99.87% accuracy at 1500 parameters after fixing MLP output-bias coordinate 0; coordinate 1 is the closest untested extension of that successful independent bias reduction.

<<<<<<< SEARCH
class TwiceAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with two reduced bias coordinates fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 2))
        return F.linear(x, weight, bias)
=======
class InteriorFiveAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0, 1, 3, 5, and 6 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(
            torch.cat((full_bias[2:3], full_bias[4:5]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middle_bias = torch.cat(
            (
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
            )
        )
        bias = self.basis @ F.pad(middle_bias, (2, 2))
        return F.linear(x, self.basis @ self.weight, bias)


class FirstTwoAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0 and 1 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (2, 0))
        return F.linear(x, self.basis @ self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = TwiceAnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = InteriorFiveAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
=======
        self.fc2 = FirstTwoAnchoredMeanZeroOutputLinear(d_ff, d_model)
>>>>>>> REPLACE