MECHANISM: Alternative hidden-unit bias anchoring

HYPOTHESIS: Fixing `fc1.bias[9]` instead of `fc1.bias[8]` will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because the four remaining hidden units are architecturally symmetric but receive different fixed-seed weight initializations.

INTENDED_EDIT: Retain learned biases for hidden units 8, 10, and 11 while reconstructing bias coordinates 0 through 7 and coordinate 9 as fixed zeros.

EVIDENCE: Fixing coordinates 0 through 7 retained 99.91% accuracy, whereas additionally fixing coordinate 8 produced 77.57%; testing a different remaining unit isolates whether that failure reflects the particular initialized row rather than insufficient three-bias capacity.

<<<<<<< SEARCH
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinates 0 through 7 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros(8)
        bias = torch.cat((zeros, self.bias_rest))
        return F.linear(x, self.weight, bias)
=======
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinates 0 through 7 and 9 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        bias = torch.cat((zeros(8), self.bias_rest[:1], zeros(1), self.bias_rest[1:]))
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE