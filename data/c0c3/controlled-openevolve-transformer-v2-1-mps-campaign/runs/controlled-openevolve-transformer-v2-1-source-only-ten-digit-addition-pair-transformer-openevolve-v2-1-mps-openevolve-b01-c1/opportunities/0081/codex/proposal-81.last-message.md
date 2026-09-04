MECHANISM: Three-row pre-MLP LayerNorm/weight quotient

HYPOTHESIS: Extending the verified `fc1` quotient from two rows to three will reduce the model from 1,544 to 1,543 parameters while retaining at least 99% accuracy, because every `fc1` row has the same redundant input-weight direction induced by the preceding bias-free LayerNorm.

INTENDED_EDIT: Constrain the final input-weight coordinate of the first three `fc1` rows and canonicalize all three rows during fresh initialization, while retaining every LayerNorm scale and linear bias.

EVIDENCE: The two-row quotient achieved 99.90% accuracy at 1,544 parameters after the one-row quotient achieved 99.76%; this directly supports one more incremental extension of the same successful weight-side constraint.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = TwoPrunedInputWeightLinear(d_model, d_ff)
=======
        self.fc1 = ThreePrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE