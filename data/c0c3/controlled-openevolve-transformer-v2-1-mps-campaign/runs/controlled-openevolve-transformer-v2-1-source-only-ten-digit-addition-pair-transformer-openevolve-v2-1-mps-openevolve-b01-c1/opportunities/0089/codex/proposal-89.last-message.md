MECHANISM: Distributed-coordinate eighth MLP quotient

HYPOTHESIS: Constraining row 7 through its first input coordinate will reduce the model to 1,538 parameters while retaining at least 99% accuracy, because it avoids further concentrating all eight quotients on the final LayerNorm coordinate.

INTENDED_EDIT: Keep rows 0–2 and 8–11 constrained on their final coordinate, but constrain row 7 on its first coordinate and canonicalize initialization to preserve its initial map on zero-mean inputs.

EVIDENCE: The 1,539-parameter seven-row model achieved 100%, while adding either row 7 or row 6 with the same final-coordinate constraint collapsed to 40.22% and 69.23%. This motivates testing whether the failure comes from excessive loss of one shared input coordinate rather than from the selected neuron alone.

<<<<<<< SEARCH
class SplitSevenPrunedInputWeightLinear(nn.Linear):
    """Linear map with seven weight scalars quotiented by zero-mean inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(full_weight[3:-4].clone())
        self.last_four_rows = nn.Parameter(full_weight[-4:, :-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        last_four_rows = F.pad(self.last_four_rows, (0, 1))
        weight = torch.cat((first_three_rows, self.weight, last_four_rows), dim=0)
        return F.linear(x, weight, self.bias)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = SplitSevenPrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = DistributedEightPrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, SplitSevenPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows without changing their maps on zero-mean inputs.
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
                module.weight.copy_(full[3:-4])
                module.last_four_rows.copy_(
                    full[-4:, :-1] - full[-4:, -1].unsqueeze(1)
                )
                nn.init.zeros_(module.bias)
=======
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
>>>>>>> REPLACE