MECHANISM: Two-row pre-MLP LayerNorm/weight quotient

HYPOTHESIS: Extending the successful `fc1` quotient to a second row will reduce the model from 1,545 to 1,544 parameters while retaining at least 99% accuracy, because each row has the same redundant weight direction induced by the preceding bias-free LayerNorm, and canonicalized initialization preserves the initial function.

INTENDED_EDIT: Constrain the final input-weight coordinate of the first two `fc1` rows, retain all LayerNorm scales and linear biases, and canonicalize both rows during fresh initialization.

EVIDENCE: The immediately preceding one-row quotient achieved 99.76% accuracy at 1,545 parameters, whereas pruning a LayerNorm scale failed at 33.43%; this supports incrementally extending the successful weight-side constraint while preserving every learned scale.

<<<<<<< SEARCH
class OnePrunedInputWeightLinear(nn.Linear):
    """Linear map with one weight scalar quotiented by zero-mean inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_row = nn.Parameter(full_weight[0, :-1].clone())
        self.weight = nn.Parameter(full_weight[1:].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_row = F.pad(self.first_row, (0, 1))
        weight = torch.cat((first_row.unsqueeze(0), self.weight), dim=0)
        return F.linear(x, weight, self.bias)
=======
class TwoPrunedInputWeightLinear(nn.Linear):
    """Linear map with two weight scalars quotiented by zero-mean inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_two_rows = nn.Parameter(full_weight[:2, :-1].clone())
        self.weight = nn.Parameter(full_weight[2:].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_two_rows = F.pad(self.first_two_rows, (0, 1))
        weight = torch.cat((first_two_rows, self.weight), dim=0)
        return F.linear(x, weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = OnePrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = TwoPrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, OnePrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, subtracting the omitted coordinate from the
            # others leaves the map unchanged on zero-mean inputs.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.first_row.copy_(full[0, :-1] - full[0, -1])
                module.weight.copy_(full[1:])
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, TwoPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, subtracting each omitted coordinate from the
            # others leaves both maps unchanged on zero-mean inputs.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.first_two_rows.copy_(
                    full[:2, :-1] - full[:2, -1].unsqueeze(1)
                )
                module.weight.copy_(full[2:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE