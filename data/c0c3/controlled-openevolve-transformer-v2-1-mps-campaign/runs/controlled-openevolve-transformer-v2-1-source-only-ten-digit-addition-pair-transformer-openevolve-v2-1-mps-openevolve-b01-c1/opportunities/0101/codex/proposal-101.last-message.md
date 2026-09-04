MECHANISM: Adjacent-neuron second-coordinate MLP quotient

HYPOTHESIS: Constraining `fc1` row 5 through its second input coordinate will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because the identical coordinate parameterization already succeeded on adjacent row 6.

INTENDED_EDIT: Remove row 5’s second input weight, reconstruct it as zero during the forward pass, and canonicalize initialization to preserve its initial map on zero-mean LayerNorm inputs.

EVIDENCE: Row 6 constrained through its second coordinate retained 99.95% accuracy, while row 5’s failures on the first, third, fourth, and fifth coordinates demonstrate strong coordinate-specific optimization effects and leave the successful second-coordinate parameterization as the most informative untested row-5 probe.

<<<<<<< SEARCH
class DistributedElevenPrunedInputWeightLinear(nn.Linear):
    """Linear map with eleven quotients distributed across input coordinates."""
=======
class DistributedTwelvePrunedInputWeightLinear(nn.Linear):
    """Linear map with twelve quotients distributed across input coordinates."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight = nn.Parameter(full_weight[5:6].clone())
=======
        self.weight = nn.Parameter(
            torch.cat((full_weight[5:6, :1], full_weight[5:6, 2:]), dim=1).clone()
        )
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
        sixth_row = torch.cat(
            (
                self.weight[:, :1],
                self.weight.new_zeros((1, 1)),
                self.weight[:, 1:],
            ),
            dim=1,
        )
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
        self.fc1 = DistributedElevenPrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = DistributedTwelvePrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, DistributedElevenPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows on the final coordinate, row three on the fourth, row
            # four on the third, row six on the second, and row seven on the
            # first coordinate.
=======
        elif isinstance(module, DistributedTwelvePrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows on the final coordinate, row three on the fourth, row
            # four on the third, rows five and six on the second, and row seven
            # on the first coordinate.
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight.copy_(full[5:6])
=======
                module.weight.copy_(
                    torch.cat(
                        (
                            full[5:6, :1] - full[5:6, 1:2],
                            full[5:6, 2:] - full[5:6, 1:2],
                        ),
                        dim=1,
                    )
                )
>>>>>>> REPLACE