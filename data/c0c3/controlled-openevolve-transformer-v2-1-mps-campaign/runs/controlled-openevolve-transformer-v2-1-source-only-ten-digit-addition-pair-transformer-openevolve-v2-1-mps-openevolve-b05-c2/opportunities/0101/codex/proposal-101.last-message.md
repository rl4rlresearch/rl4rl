MECHANISM: Final high-side MLP output-bias anchoring

HYPOTHESIS: Fixing reduced MLP output-bias coordinate 6 while preserving sensitive coordinates 3 and 4 will yield 1496 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Reconstruct MLP output-bias coordinates 0, 1, 2, 5, and 6 as zero, leaving only coordinates 3 and 4 learned.

EVIDENCE: The current 1497-parameter design achieved 99.81% accuracy after fixing coordinate 5, while fixing coordinates 3 or 4 caused failure; coordinate 6 is the only remaining untested bias coordinate whose removal preserves both sensitive coordinates.

<<<<<<< SEARCH
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
=======
class FirstThreeAndLastTwoAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0, 1, 2, 5, and 6 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[3:5])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (3, 2))
        return F.linear(x, self.basis @ self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = FirstThreeAndCoordinateFiveAnchoredMeanZeroOutputLinear(
            d_ff, d_model
        )
=======
        self.fc2 = FirstThreeAndLastTwoAnchoredMeanZeroOutputLinear(
            d_ff, d_model
        )
>>>>>>> REPLACE