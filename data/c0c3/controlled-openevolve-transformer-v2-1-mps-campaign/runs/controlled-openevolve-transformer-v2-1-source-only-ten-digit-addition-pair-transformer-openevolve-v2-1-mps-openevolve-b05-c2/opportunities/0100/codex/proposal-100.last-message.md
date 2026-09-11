MECHANISM: Nonadjacent high-side MLP output-bias anchoring

HYPOTHESIS: Fixing reduced MLP output-bias coordinate 5 while preserving coordinates 3 and 4 will yield 1497 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified 1498-parameter attention and MLP bias pattern, then reconstruct MLP output-bias coordinates 0, 1, 2, and 5 as zero while learning coordinates 3, 4, and 6.

EVIDENCE: The 1498-parameter design achieved 99.94% accuracy with MLP bias coordinates 0–2 fixed; additionally fixing coordinates 3 or 4 failed, so coordinate 5 is the closest untested reduction that preserves both empirically sensitive coordinates.

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


class FirstThreeAndCoordinateFiveAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0, 1, 2, and 5 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(
            torch.cat((full_bias[3:5], full_bias[6:]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced_bias = torch.cat(
            (
                self.bias[:2],
                self.bias.new_zeros(1),
                self.bias[2:],
            )
        )
        bias = self.basis @ F.pad(reduced_bias, (3, 0))
        return F.linear(x, self.basis @ self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = LowSplitFourAnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = InteriorFiveAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
=======
        self.fc2 = FirstThreeAndCoordinateFiveAnchoredMeanZeroOutputLinear(
            d_ff, d_model
        )
>>>>>>> REPLACE