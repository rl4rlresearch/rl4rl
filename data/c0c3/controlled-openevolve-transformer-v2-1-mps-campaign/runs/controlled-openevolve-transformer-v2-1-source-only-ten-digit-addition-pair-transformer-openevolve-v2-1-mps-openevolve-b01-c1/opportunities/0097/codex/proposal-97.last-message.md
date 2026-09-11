MECHANISM: First-coordinate twelfth MLP quotient

HYPOTHESIS: Constraining `fc1` row 5 through its first input coordinate will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because the first-coordinate parameterization previously rescued row 7 and avoids row 5’s unsuccessful third-, fourth-, and fifth-coordinate choices.

INTENDED_EDIT: Remove row 5’s first input weight, reconstruct it as zero during the forward pass, and canonicalize fresh initialization to preserve its initial map on zero-mean LayerNorm inputs.

EVIDENCE: Row 5 achieved only 66.51%, 37.9%, and 9.93% when constrained through coordinates three, four, and five, while moving row 7’s constraint to the first coordinate raised accuracy from 40.22% to 99.95%; this makes the first coordinate the strongest untested row-5 alternative.

<<<<<<< SEARCH
class DistributedElevenPrunedInputWeightLinear(nn.Linear):
    """Linear map with eleven quotients distributed across input coordinates."""
=======
class DistributedElevenPrunedInputWeightLinear(nn.Linear):
    """Linear map with twelve quotients distributed across input coordinates."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight = nn.Parameter(full_weight[5:6].clone())
=======
        self.weight = nn.Parameter(full_weight[5, 1:].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        fifth_row = torch.cat(
            (
                self.fifth_row[:2],
                self.fifth_row.new_zeros(1),
                self.fifth_row[2:],
            )
        ).unsqueeze(0)
        last_four_rows = F.pad(self.last_four_rows, (0, 1))
=======
        fifth_row = torch.cat(
            (
                self.fifth_row[:2],
                self.fifth_row.new_zeros(1),
                self.fifth_row[2:],
            )
        ).unsqueeze(0)
        sixth_row = F.pad(self.weight, (1, 0)).unsqueeze(0)
        last_four_rows = F.pad(self.last_four_rows, (0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fifth_row,
                self.weight,
                seventh_row,
=======
                fifth_row,
                sixth_row,
                seventh_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
            # four on the third, row six on the second, and row seven on the
            # first coordinate.
=======
            # four on the third, row six on the second, and rows five and seven
            # on the first coordinate.
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight.copy_(full[5:6])
=======
                module.weight.copy_(full[5, 1:] - full[5, 0])
>>>>>>> REPLACE