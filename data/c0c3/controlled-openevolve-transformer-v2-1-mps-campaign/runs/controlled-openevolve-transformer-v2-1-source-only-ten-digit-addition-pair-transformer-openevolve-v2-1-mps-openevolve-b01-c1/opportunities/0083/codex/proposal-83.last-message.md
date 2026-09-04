MECHANISM: Alternate-neuron fourth MLP weight quotient

HYPOTHESIS: Constraining the last `fc1` row instead of the fourth row will produce a 1,542-parameter model with at least 99% accuracy, showing whether the previous collapse was specific to which initialized neuron received the fourth constraint.

INTENDED_EDIT: Retain the successful constraints on rows 0–2, additionally remove the final input-weight coordinate from row 11, and canonicalize all four affected rows while preserving the freshly initialized function.

EVIDENCE: Three consecutive constrained rows achieved 100% accuracy at 1,543 parameters, while adding row 3 collapsed to 18.65%; moving the fourth constraint to a different, otherwise exchangeable MLP neuron is the smallest test of a row-specific optimization failure.

<<<<<<< SEARCH
class ThreePrunedInputWeightLinear(nn.Linear):
    """Linear map with three weight scalars quotiented by zero-mean inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.weight = nn.Parameter(full_weight[3:].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        weight = torch.cat((first_three_rows, self.weight), dim=0)
        return F.linear(x, weight, self.bias)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = ThreePrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = SplitFourPrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ThreePrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, subtracting each omitted coordinate from the
            # others leaves all three maps unchanged on zero-mean inputs.
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
                module.weight.copy_(full[3:])
                nn.init.zeros_(module.bias)
=======
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
>>>>>>> REPLACE