MECHANISM: Coordinate-alternate eleventh MLP quotient

HYPOTHESIS: Constraining `fc1` row 5 through its fourth input coordinate will reduce the model from 1,536 to 1,535 parameters while retaining at least 99% accuracy, because changing the constrained coordinate previously rescued row 7 and avoids reusing row 5’s failed third-coordinate constraint.

INTENDED_EDIT: Remove row 5’s fourth input weight, reconstruct it as zero during the forward pass, and canonicalize fresh initialization to preserve the row’s initial map on zero-mean LayerNorm inputs.

EVIDENCE: Row 5 constrained on the third coordinate collapsed to 66.51%, while row 7 improved from 40.22% with a final-coordinate constraint to 99.95% with a first-coordinate constraint. This directly motivates testing whether a different distributed coordinate similarly rescues row 5.

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
class DistributedTenPrunedInputWeightLinear(nn.Linear):
    """Linear map with eleven quotients distributed across input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(full_weight[3:4].clone())
        self.fifth_row = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.seventh_row = nn.Parameter(
            torch.cat((full_weight[6, :1], full_weight[6, 2:])).clone()
        )
=======
        self.sixth_row = nn.Parameter(
            torch.cat((full_weight[5, :3], full_weight[5, 4:])).clone()
        )
        self.seventh_row = nn.Parameter(
            torch.cat((full_weight[6, :1], full_weight[6, 2:])).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        seventh_row = torch.cat(
            (
                self.seventh_row[:1],
                self.seventh_row.new_zeros(1),
                self.seventh_row[1:],
            )
        ).unsqueeze(0)
=======
        sixth_row = torch.cat(
            (
                self.sixth_row[:3],
                self.sixth_row.new_zeros(1),
                self.sixth_row[3:],
            )
        ).unsqueeze(0)
        seventh_row = torch.cat(
            (
                self.seventh_row[:1],
                self.seventh_row.new_zeros(1),
                self.seventh_row[1:],
            )
        ).unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
                fifth_row,
                self.weight[1:],
                seventh_row,
=======
                fifth_row,
                sixth_row,
                seventh_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
            # four rows on the final coordinate, row four on the third, row six
            # on the second coordinate, and row seven on the first.
=======
            # four rows on the final coordinate, row four on the third, row five
            # on the fourth, row six on the second, and row seven on the first.
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight.copy_(
                    torch.cat((full[3:4], full[5:6]), dim=0)
                )
=======
                module.weight.copy_(full[3:4])
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.seventh_row.copy_(
                    torch.cat(
                        (
                            full[6, :1] - full[6, 1],
=======
                module.sixth_row.copy_(
                    torch.cat(
                        (
                            full[5, :3] - full[5, 3],
                            full[5, 4:] - full[5, 3],
                        )
                    )
                )
                module.seventh_row.copy_(
                    torch.cat(
                        (
                            full[6, :1] - full[6, 1],
>>>>>>> REPLACE