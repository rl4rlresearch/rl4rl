MECHANISM: Independent MLP hidden-bias anchoring

HYPOTHESIS: Starting from the qualified 1497-parameter design, fixing MLP input-bias coordinate 0 will yield 1496 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Preserve the successful MLP output-bias anchors at coordinates 0, 1, 2, and 5, and independently reconstruct the first of twelve MLP hidden biases as zero.

EVIDENCE: Reference Design 3 achieved 99.81% accuracy at 1497 parameters, while removing its remaining output-bias coordinate 6 failed at 27.87%; prior successful reductions moved to a distinct zero-initialized affine bias after exhausting a sensitive bias family, motivating the larger MLP input-bias vector as the next independent target.

<<<<<<< SEARCH
class FirstThreeAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0, 1, and 2 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[3:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (3, 0))
        return F.linear(x, self.basis @ self.weight, bias)
=======
class FirstAnchoredMeanZeroInputLinear(MeanZeroInputLinear):
    """Input map with hidden-bias coordinate 0 fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight @ self.basis.transpose(0, 1)
        return F.linear(x, weight, F.pad(self.bias, (1, 0)))


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
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = FirstThreeAnchoredMeanZeroOutputLinear(d_ff, d_model)
=======
        self.fc1 = FirstAnchoredMeanZeroInputLinear(d_model, d_ff)
        self.fc2 = FirstThreeAndCoordinateFiveAnchoredMeanZeroOutputLinear(
            d_ff, d_model
        )
>>>>>>> REPLACE