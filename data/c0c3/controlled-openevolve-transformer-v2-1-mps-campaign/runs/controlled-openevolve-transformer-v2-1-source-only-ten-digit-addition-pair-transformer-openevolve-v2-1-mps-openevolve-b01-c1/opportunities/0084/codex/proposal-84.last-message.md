MECHANISM: Tail-neuron fifth MLP weight quotient

HYPOTHESIS: Constraining the penultimate `fc1` row alongside rows 0–2 and 11 will reduce the model to 1,541 parameters while retaining at least 99% accuracy, indicating that the successful fourth constraint is characteristic of tail MLP neurons rather than unique to row 11.

INTENDED_EDIT: Remove the final input-weight coordinate from `fc1` row 10 and canonicalize its fresh initialization, leaving all previously successful constraints and training settings unchanged.

EVIDENCE: Rows 0–2 achieved 100% accuracy at 1,543 parameters; adding row 3 collapsed to 18.65%, whereas constraining row 11 instead achieved 99.85% at 1,542 parameters. Testing adjacent row 10 is the most targeted next probe of this row-specific optimization effect.

<<<<<<< SEARCH
class SplitFourPrunedInputWeightLinear(nn.Linear):
    """Linear map with four weight scalars quotiented by zero-mean inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(full_weight[3:-1].clone())
        self.last_row = nn.Parameter(full_weight[-1, :-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        last_row = F.pad(self.last_row, (0, 1)).unsqueeze(0)
        weight = torch.cat((first_three_rows, self.weight, last_row), dim=0)
        return F.linear(x, weight, self.bias)
=======
class SplitFivePrunedInputWeightLinear(nn.Linear):
    """Linear map with five weight scalars quotiented by zero-mean inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(full_weight[3:-2].clone())
        self.last_two_rows = nn.Parameter(full_weight[-2:, :-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        last_two_rows = F.pad(self.last_two_rows, (0, 1))
        weight = torch.cat((first_three_rows, self.weight, last_two_rows), dim=0)
        return F.linear(x, weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = SplitFourPrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = SplitFivePrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, SplitFourPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # row without changing their maps on zero-mean inputs.
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
                module.weight.copy_(full[3:-1])
                module.last_row.copy_(full[-1, :-1] - full[-1, -1])
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, SplitFivePrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # two rows without changing their maps on zero-mean inputs.
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
                module.weight.copy_(full[3:-2])
                module.last_two_rows.copy_(
                    full[-2:, :-1] - full[-2:, -1].unsqueeze(1)
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE