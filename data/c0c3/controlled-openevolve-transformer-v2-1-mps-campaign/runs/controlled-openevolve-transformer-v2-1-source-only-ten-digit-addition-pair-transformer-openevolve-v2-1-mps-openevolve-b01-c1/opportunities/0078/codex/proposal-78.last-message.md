MECHANISM: Single-row pre-MLP LayerNorm/weight quotient

HYPOTHESIS: Removing one `fc1` input-weight scalar will reduce the model from 1,546 to 1,545 parameters while retaining at least 99% accuracy, because zero-mean LayerNorm outputs make one weight direction per MLP neuron functionally redundant while all eight learned `ln2` scales remain trainable.

INTENDED_EDIT: Constrain one `fc1` row to have a fixed final coordinate, canonicalize its fresh initialization to preserve the initial function, and leave the other eleven rows unchanged.

EVIDENCE: Fixing an `ln2` scale coordinate failed at 33.43%, showing that scale optimization is load-bearing; this patch preserves every scale and instead removes one scalar from the opposite side of the same exact LayerNorm/linear quotient. The successful complete `ln2` bias quotient further supports targeting redundancy at this interface.

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = OnePrunedInputWeightLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreeResidualLinear):
=======
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
        elif isinstance(module, MeanFreeResidualLinear):
>>>>>>> REPLACE