MECHANISM: Staircase-distributed tenth MLP quotient

HYPOTHESIS: Constraining `fc1` row 5 through its third input coordinate will reduce the model from 1,537 to 1,536 parameters while retaining at least 99% accuracy, because it extends the successful coordinate-distribution pattern without concentrating another quotient on previously used coordinates.

INTENDED_EDIT: Preserve the nine verified constraints, remove row 5’s third input weight, and canonicalize its fresh initialization to preserve its initial map on zero-mean LayerNorm inputs.

EVIDENCE: Adding row 6 on the second coordinate retained 99.95% after adding row 7 on the first coordinate also retained 99.95%, whereas concentrating additional constraints on the final coordinate caused severe collapses; row 5 on the third coordinate is the smallest targeted continuation.

<<<<<<< SEARCH
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
=======
class DistributedTenPrunedInputWeightLinear(nn.Linear):
    """Linear map with ten quotients distributed across input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(full_weight[3:5].clone())
        self.sixth_row = nn.Parameter(
            torch.cat((full_weight[5, :2], full_weight[5, 3:])).clone()
        )
        self.seventh_row = nn.Parameter(
            torch.cat((full_weight[6, :1], full_weight[6, 2:])).clone()
        )
        self.eighth_row = nn.Parameter(full_weight[7, 1:].clone())
        self.last_four_rows = nn.Parameter(full_weight[-4:, :-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        sixth_row = torch.cat(
            (
                self.sixth_row[:2],
                self.sixth_row.new_zeros(1),
                self.sixth_row[2:],
            )
        ).unsqueeze(0)
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
                sixth_row,
                seventh_row,
                eighth_row,
                last_four_rows,
            ),
            dim=0,
        )
        return F.linear(x, weight, self.bias)
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
=======
        elif isinstance(module, DistributedTenPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows on the final coordinate, row five on the third
            # coordinate, row six on the second, and row seven on the first.
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
                module.weight.copy_(full[3:5])
                module.sixth_row.copy_(
                    torch.cat(
                        (
                            full[5, :2] - full[5, 2],
                            full[5, 3:] - full[5, 2],
                        )
                    )
                )
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