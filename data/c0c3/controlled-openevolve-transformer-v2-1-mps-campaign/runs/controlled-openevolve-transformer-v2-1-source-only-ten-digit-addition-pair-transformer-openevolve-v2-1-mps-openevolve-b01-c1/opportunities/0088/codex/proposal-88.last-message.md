MECHANISM: Alternate-neuron eighth MLP weight quotient

HYPOTHESIS: Constraining `fc1` row 6 while leaving the failed row 7 unconstrained will reduce the model from 1,539 to 1,538 parameters while retaining at least 99% accuracy, demonstrating another row-specific optimization effect.

INTENDED_EDIT: Preserve constraints on rows 0–2 and 8–11, additionally remove row 6’s final input-weight coordinate, and canonicalize its fresh initialization without changing its initial map on zero-mean inputs.

EVIDENCE: The seven-row model achieved 100% accuracy, whereas adding row 7 collapsed to 40.22%; earlier, moving a failed fourth constraint from row 3 to row 11 restored 99.85%, directly motivating another alternate-row probe.

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


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = SplitSevenPrunedInputWeightLinear(d_model, d_ff)
=======
class SplitEightPrunedInputWeightLinear(nn.Linear):
    """Linear map with eight weight scalars quotiented by zero-mean inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.middle_three_rows = nn.Parameter(full_weight[3:6].clone())
        self.row_six = nn.Parameter(full_weight[6:7, :-1].clone())
        self.weight = nn.Parameter(full_weight[7:-4].clone())
        self.last_four_rows = nn.Parameter(full_weight[-4:, :-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        row_six = F.pad(self.row_six, (0, 1))
        last_four_rows = F.pad(self.last_four_rows, (0, 1))
        weight = torch.cat(
            (
                first_three_rows,
                self.middle_three_rows,
                row_six,
                self.weight,
                last_four_rows,
            ),
            dim=0,
        )
        return F.linear(x, weight, self.bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = SplitEightPrunedInputWeightLinear(d_model, d_ff)
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
        elif isinstance(module, SplitEightPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two, row six, and
            # the last four rows without changing their zero-mean-input maps.
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
                module.middle_three_rows.copy_(full[3:6])
                module.row_six.copy_(
                    full[6:7, :-1] - full[6:7, -1]
                )
                module.weight.copy_(full[7:-4])
                module.last_four_rows.copy_(
                    full[-4:, :-1] - full[-4:, -1].unsqueeze(1)
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE