MECHANISM: LayerNorm zero-sum input-weight quotient

HYPOTHESIS: Removing one `fc1` input-weight scalar will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because `ln2`’s normalized activations lie in a zero-sum hyperplane and the first neuron’s retained bias absorbs the affine offset.

INTENDED_EDIT: Reparameterize the first `fc1` weight row with its final coordinate fixed at zero, while preserving initialization outputs, RNG consumption, all six learned `ln2` bias coordinates, and the successful one-pruned `fc1` output bias.

EVIDENCE: The 1,554-parameter design achieved 99.92%, while removing a second `fc1` bias fell to 84.48% and fixing an `ln1` scale fell to 86.89%; this motivates removing a distinct exact redundancy that retains both the downstream neuron bias and every LayerNorm scale and bias coordinate.

<<<<<<< SEARCH
class OnePrunedOutputBiasLinear(nn.Linear):
    """Linear layer with one bias scalar absorbed by the preceding LayerNorm bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.linear(x, self.weight, bias)
=======
class OnePrunedOutputBiasLinear(nn.Linear):
    """Linear layer quotienting one normalized-input weight and one output bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        full_bias = self.bias.detach()
        self.first_weight = nn.Parameter(
            (full_weight[0, :-1] - full_weight[0, -1]).clone()
        )
        self.weight = nn.Parameter(full_weight[1:].clone())
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (0, 1)).unsqueeze(0)
        weight = torch.cat((first_row, self.weight), dim=0)
        bias = F.pad(self.bias, (0, 1))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreeResidualLinear):
            # Draw the original full output matrix so subsequent initialization
            # keeps the same RNG sequence, then retain its observable component.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(module.bias_basis.transpose(0, 1) @ full)
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        elif isinstance(module, MeanFreeResidualLinear):
            # Draw the original full output matrix so subsequent initialization
            # keeps the same RNG sequence, then retain its observable component.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(module.bias_basis.transpose(0, 1) @ full)
                nn.init.zeros_(module.bias)
        elif isinstance(module, OnePrunedOutputBiasLinear):
            # Preserve the original full draw. At unit LayerNorm scale and zero
            # bias, subtracting the final coefficient from the other coefficients
            # leaves the first neuron's output unchanged on zero-sum inputs.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.first_weight.copy_(full[0, :-1] - full[0, -1])
                module.weight.copy_(full[1:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE