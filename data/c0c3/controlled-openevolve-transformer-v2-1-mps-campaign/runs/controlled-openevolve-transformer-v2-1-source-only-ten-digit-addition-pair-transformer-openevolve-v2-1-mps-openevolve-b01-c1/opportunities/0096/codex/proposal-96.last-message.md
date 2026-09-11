MECHANISM: Coordinate-alternate twelfth MLP quotient

HYPOTHESIS: Constraining `fc1` row 5 through its fifth input coordinate will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because coordinate choice has shown strong neuron-specific effects and this coordinate remains untested for row 5.

INTENDED_EDIT: Remove row 5’s fifth input weight, reconstruct it as zero during the forward pass, and canonicalize fresh initialization to preserve the row’s initial map on zero-mean LayerNorm inputs.

EVIDENCE: Row 5 failed when constrained on its third and fourth coordinates, but moving row 7’s constraint from its final coordinate to its first rescued accuracy from 40.22% to 99.95%. This makes another coordinate-specific row-5 probe the smallest informative extension of the 1,535-parameter design.

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
            torch.cat((full_weight[5, :4], full_weight[5, 5:])).clone()
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
                self.weight[:4],
                self.weight.new_zeros(1),
                self.weight[4:],
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
            # four on the third, row five on the fifth, row six on the second,
            # and row seven on the first coordinate.
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight.copy_(full[5:6])
=======
                module.weight.copy_(
                    torch.cat(
                        (
                            full[5, :4] - full[5, 4],
                            full[5, 5:] - full[5, 4],
                        )
                    )
                )
>>>>>>> REPLACE