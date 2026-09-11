MECHANISM: Alternate-neuron staircase MLP quotient

HYPOTHESIS: Constraining `fc1` row 3 through its fourth input coordinate will reduce the model from 1,536 to 1,535 parameters while retaining at least 99% accuracy, showing that row 5’s failure on this coordinate was neuron-specific.

INTENDED_EDIT: Remove row 3’s fourth input weight, reconstruct it as zero during the forward pass, and canonicalize its fresh initialization to preserve its initial map on zero-mean LayerNorm inputs.

EVIDENCE: Row 5 constrained on its third coordinate failed at 66.51%, while moving that same constraint to row 4 achieved 99.41%; row 5’s fourth-coordinate failure at 37.9% therefore most directly motivates testing the same coordinate on the remaining unconstrained row 3.

<<<<<<< SEARCH
class DistributedTenPrunedInputWeightLinear(nn.Linear):
    """Linear map with ten quotients distributed across input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(
            torch.cat((full_weight[3:4], full_weight[5:6]), dim=0).clone()
        )
        self.fifth_row = nn.Parameter(
=======
class DistributedElevenPrunedInputWeightLinear(nn.Linear):
    """Linear map with eleven quotients distributed across input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.fourth_row = nn.Parameter(
            torch.cat((full_weight[3, :3], full_weight[3, 4:])).clone()
        )
        self.weight = nn.Parameter(full_weight[5:6].clone())
        self.fifth_row = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        seventh_row = torch.cat(
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        fourth_row = torch.cat(
            (
                self.fourth_row[:3],
                self.fourth_row.new_zeros(1),
                self.fourth_row[3:],
            )
        ).unsqueeze(0)
        seventh_row = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                first_three_rows,
                self.weight[:1],
                fifth_row,
                self.weight[1:],
                seventh_row,
=======
                first_three_rows,
                fourth_row,
                fifth_row,
                self.weight,
                seventh_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = DistributedTenPrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = DistributedElevenPrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, DistributedTenPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows on the final coordinate, row four on the third, row six
            # on the second coordinate, and row seven on the first.
=======
        elif isinstance(module, DistributedElevenPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows on the final coordinate, row three on the fourth, row
            # four on the third, row six on the second, and row seven on the
            # first coordinate.
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight.copy_(
                    torch.cat((full[3:4], full[5:6]), dim=0)
                )
                module.fifth_row.copy_(
=======
                module.fourth_row.copy_(
                    torch.cat(
                        (
                            full[3, :3] - full[3, 3],
                            full[3, 4:] - full[3, 3],
                        )
                    )
                )
                module.weight.copy_(full[5:6])
                module.fifth_row.copy_(
>>>>>>> REPLACE