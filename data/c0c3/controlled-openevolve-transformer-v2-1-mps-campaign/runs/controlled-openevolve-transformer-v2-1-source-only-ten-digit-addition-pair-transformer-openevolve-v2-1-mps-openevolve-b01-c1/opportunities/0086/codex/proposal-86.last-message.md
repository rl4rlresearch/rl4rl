MECHANISM: Tail-neuron seventh MLP weight quotient

HYPOTHESIS: Constraining `fc1` row 8 alongside rows 0–2 and 9–11 will reduce the model to 1,539 parameters while retaining at least 99% accuracy, showing that the tail-neuron optimization tolerance extends through the final four rows.

INTENDED_EDIT: Remove the final input-weight coordinate from `fc1` row 8 and canonicalize its fresh initialization, preserving all existing model and training pathways.

EVIDENCE: Constraints on rows 0–2 reached 100% at 1,543 parameters; row 3 caused collapse, while successively adding tail rows 11, 10, and 9 retained 99.85%, 99.96%, and 99.89%. Row 8 is the most informative next boundary probe.

<<<<<<< SEARCH
class SplitSixPrunedInputWeightLinear(nn.Linear):
    """Linear map with six weight scalars quotiented by zero-mean inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(full_weight[3:-3].clone())
        self.last_three_rows = nn.Parameter(full_weight[-3:, :-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        last_three_rows = F.pad(self.last_three_rows, (0, 1))
        weight = torch.cat((first_three_rows, self.weight, last_three_rows), dim=0)
        return F.linear(x, weight, self.bias)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = SplitSixPrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = SplitSevenPrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, SplitSixPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # three rows without changing their maps on zero-mean inputs.
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
                module.weight.copy_(full[3:-3])
                module.last_three_rows.copy_(
                    full[-3:, :-1] - full[-3:, -1].unsqueeze(1)
                )
                nn.init.zeros_(module.bias)
=======
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
>>>>>>> REPLACE