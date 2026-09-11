MECHANISM: Nonadjacent MLP output-bias anchoring

HYPOTHESIS: Fixing reduced MLP output-bias coordinate 4 while retaining the successful anchors at coordinates 0–2 will yield 1497 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified 1498-parameter MLP bias reduction, then reconstruct reduced bias coordinates 0, 1, 2, and 4 as zero while learning coordinates 3, 5, and 6.

EVIDENCE: Fixing MLP bias coordinates 0–2 achieved 99.94% accuracy at 1498 parameters, whereas additionally fixing coordinate 3 fell to 45.34%; testing coordinate 4 distinguishes coordinate sensitivity from a general 1497-parameter capacity limit.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class FirstThreeAndCoordinateFourAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0, 1, 2, and 4 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(
            torch.cat((full_bias[3:4], full_bias[5:]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tail_bias = torch.cat(
            (
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
            )
        )
        bias = self.basis @ F.pad(tail_bias, (3, 0))
        return F.linear(x, self.basis @ self.weight, bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
=======
        self.fc2 = FirstThreeAndCoordinateFourAnchoredMeanZeroOutputLinear(
            d_ff, d_model
        )
>>>>>>> REPLACE