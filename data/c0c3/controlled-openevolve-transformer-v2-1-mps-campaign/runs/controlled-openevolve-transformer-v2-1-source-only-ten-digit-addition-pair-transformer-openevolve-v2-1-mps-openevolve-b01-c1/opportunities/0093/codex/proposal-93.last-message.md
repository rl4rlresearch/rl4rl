MECHANISM: Alternate-neuron tenth MLP quotient

HYPOTHESIS: Constraining `fc1` row 4 through its third input coordinate will reduce the model from 1,537 to 1,536 parameters while retaining at least 99% accuracy, showing that the failed row-5 constraint was another row-specific optimization failure rather than a limit on distributed quotients.

INTENDED_EDIT: Leave row 5 unconstrained, remove row 4’s third input weight, and canonicalize its fresh initialization while preserving its initial map on zero-mean LayerNorm inputs.

EVIDENCE: Moving the failed fourth final-coordinate constraint from row 3 to row 11 restored accuracy from 18.65% to 99.85%, establishing strong neuron-specific optimization effects. Since row 5 on the third coordinate reached only 66.51%, testing the adjacent exchangeable row on the same coordinate is the most targeted next probe.

<<<<<<< SEARCH
class DistributedNinePrunedInputWeightLinear(nn.Linear):
    """Linear map with nine quotients distributed across input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(full_weight[3:6].clone())
        self.seventh_row = nn.Parameter(
=======
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
            torch.cat((full_weight[4, :2], full_weight[4, 3:])).clone()
        )
        self.seventh_row = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
        eighth_row = F.pad(self.eighth_row, (1, 0)).unsqueeze(0)
        last_four_rows = F.pad(self.last_four_rows, (0, 1))
        weight = torch.cat(
            (
                first_three_rows,
                self.weight,
                seventh_row,
=======
        eighth_row = F.pad(self.eighth_row, (1, 0)).unsqueeze(0)
        fifth_row = torch.cat(
            (
                self.fifth_row[:2],
                self.fifth_row.new_zeros(1),
                self.fifth_row[2:],
            )
        ).unsqueeze(0)
        last_four_rows = F.pad(self.last_four_rows, (0, 1))
        weight = torch.cat(
            (
                first_three_rows,
                self.weight[:1],
                fifth_row,
                self.weight[1:],
                seventh_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = DistributedNinePrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = DistributedTenPrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, DistributedNinePrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows on the final coordinate, row six on the second
            # coordinate, and row seven on the first.
=======
        elif isinstance(module, DistributedTenPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows on the final coordinate, row four on the third, row six
            # on the second coordinate, and row seven on the first.
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight.copy_(full[3:6])
                module.seventh_row.copy_(
=======
                module.weight.copy_(
                    torch.cat((full[3:4], full[5:6]), dim=0)
                )
                module.fifth_row.copy_(
                    torch.cat(
                        (
                            full[4, :2] - full[4, 2],
                            full[4, 3:] - full[4, 2],
                        )
                    )
                )
                module.seventh_row.copy_(
>>>>>>> REPLACE