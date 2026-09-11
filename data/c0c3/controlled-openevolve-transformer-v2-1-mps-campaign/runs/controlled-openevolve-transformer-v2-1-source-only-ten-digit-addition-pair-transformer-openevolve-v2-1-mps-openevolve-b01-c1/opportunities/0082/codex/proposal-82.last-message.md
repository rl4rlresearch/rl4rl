MECHANISM: Four-row pre-MLP LayerNorm/weight quotient

HYPOTHESIS: Extending the verified `fc1` quotient from three rows to four will reduce the model from 1,543 to 1,542 parameters while retaining at least 99% accuracy, because each `fc1` row has the same redundant input-weight direction induced by the preceding bias-free LayerNorm.

INTENDED_EDIT: Constrain the final input-weight coordinate of the first four `fc1` rows and canonicalize all four rows during fresh initialization, while retaining every LayerNorm scale and linear bias.

EVIDENCE: The same incremental quotient achieved 99.76% with one row, 99.90% with two rows, and 100% with three rows; this directly motivates testing one additional row without modifying any pathway implicated by prior failures.

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
class FourPrunedInputWeightLinear(nn.Linear):
    """Linear map with four weight scalars quotiented by zero-mean inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_four_rows = nn.Parameter(full_weight[:4, :-1].clone())
        self.weight = nn.Parameter(full_weight[4:].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_four_rows = F.pad(self.first_four_rows, (0, 1))
        weight = torch.cat((first_four_rows, self.weight), dim=0)
        return F.linear(x, weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = ThreePrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = FourPrunedInputWeightLinear(d_model, d_ff)
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
        elif isinstance(module, FourPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, subtracting each omitted coordinate from the
            # others leaves all four maps unchanged on zero-mean inputs.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.first_four_rows.copy_(
                    full[:4, :-1] - full[:4, -1].unsqueeze(1)
                )
                module.weight.copy_(full[4:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE