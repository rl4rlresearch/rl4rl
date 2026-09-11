MECHANISM: Distributed-coordinate ninth MLP quotient

HYPOTHESIS: Constraining `fc1` row 6 through its second input coordinate will reduce the model from 1,538 to 1,537 parameters while retaining at least 99% accuracy, because it adds an exact LayerNorm-induced quotient without further concentrating constraints on either the final or first coordinate.

INTENDED_EDIT: Keep the eight verified constraints, remove row 6’s second input weight, and canonicalize its fresh initialization to preserve its initial map on zero-mean inputs.

EVIDENCE: Eight constraints concentrated on the final coordinate failed for both row 7 and row 6, while moving row 7’s constraint to the first coordinate achieved 99.95%; distributing row 6’s new constraint onto another coordinate is the smallest targeted extension of that successful mechanism.

<<<<<<< SEARCH
class DistributedEightPrunedInputWeightLinear(nn.Linear):
    """Linear map with eight quotients distributed across input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(full_weight[3:7].clone())
        self.eighth_row = nn.Parameter(full_weight[7, 1:].clone())
        self.last_four_rows = nn.Parameter(full_weight[-4:, :-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        eighth_row = F.pad(self.eighth_row, (1, 0)).unsqueeze(0)
        last_four_rows = F.pad(self.last_four_rows, (0, 1))
        weight = torch.cat(
            (first_three_rows, self.weight, eighth_row, last_four_rows), dim=0
        )
        return F.linear(x, weight, self.bias)
=======
class DistributedNinePrunedInputWeightLinear(nn.Linear):
    """Linear map with nine quotients distributed across input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(full_weight[3:6].clone())
        self.seventh_row = nn.Parameter(
            torch.cat((full_weight[6, :1], full_weight[6, 2:])).clone()
        )
        self.eighth_row = nn.Parameter(full_weight[7, 1:].clone())
        self.last_four_rows = nn.Parameter(full_weight[-4:, :-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        seventh_row = torch.cat(
            (
                self.seventh_row[:1],
                self.seventh_row.new_zeros(1),
                self.seventh_row[1:],
            )
        ).unsqueeze(0)
        eighth_row = F.pad(self.eighth_row, (1, 0)).unsqueeze(0)
        last_four_rows = F.pad(self.last_four_rows, (0, 1))
        weight = torch.cat(
            (
                first_three_rows,
                self.weight,
                seventh_row,
                eighth_row,
                last_four_rows,
            ),
            dim=0,
        )
        return F.linear(x, weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = DistributedEightPrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = DistributedNinePrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, DistributedEightPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows on the final coordinate, and row seven on the first.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.first_three_rows.copy_(
                    full[:3, :-1] - full[:3, -1].unsqueeze(1)
                )
                module.weight.copy_(full[3:7])
                module.eighth_row.copy_(full[7, 1:] - full[7, 0])
                module.last_four_rows.copy_(
                    full[-4:, :-1] - full[-4:, -1].unsqueeze(1)
                )
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, DistributedNinePrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows on the final coordinate, row six on the second
            # coordinate, and row seven on the first.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.first_three_rows.copy_(
                    full[:3, :-1] - full[:3, -1].unsqueeze(1)
                )
                module.weight.copy_(full[3:6])
                module.seventh_row.copy_(
                    torch.cat(
                        (
                            full[6, :1] - full[6, 1],
                            full[6, 2:] - full[6, 1],
                        )
                    )
                )
                module.eighth_row.copy_(full[7, 1:] - full[7, 0])
                module.last_four_rows.copy_(
                    full[-4:, :-1] - full[-4:, -1].unsqueeze(1)
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE