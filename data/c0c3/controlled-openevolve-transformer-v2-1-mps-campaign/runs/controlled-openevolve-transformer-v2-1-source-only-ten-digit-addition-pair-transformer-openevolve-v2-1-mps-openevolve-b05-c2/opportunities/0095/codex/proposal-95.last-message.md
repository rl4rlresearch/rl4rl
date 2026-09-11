MECHANISM: Independent MLP output-bias coordinate anchoring

HYPOTHESIS: Reproducing the qualified 1501-parameter attention-bias pattern and fixing reduced MLP output-bias coordinate 0 will yield 1500 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Keep only attention projection-bias coordinates 2 and 4, then reconstruct MLP output-bias coordinate 0 as zero while learning the remaining six coordinates.

EVIDENCE: Reference Design 2 achieved 99.96% accuracy at 1501 parameters with attention-bias coordinates 0, 1, 3, 5, and 6 fixed; further attention-bias anchoring failed, motivating a distinct zero-initialized affine-bias reduction in the MLP output.

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


class FirstAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinate 0 fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (1, 0))
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
        self.fc2 = FirstAnchoredMeanZeroOutputLinear(d_ff, d_model)
>>>>>>> REPLACE