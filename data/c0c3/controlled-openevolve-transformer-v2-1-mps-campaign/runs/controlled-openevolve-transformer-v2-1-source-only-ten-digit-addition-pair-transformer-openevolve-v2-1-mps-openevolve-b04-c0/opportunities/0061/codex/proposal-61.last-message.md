MECHANISM: Third alternative hidden-unit bias anchoring

HYPOTHESIS: Fixing `fc1.bias[11]` in addition to coordinates 0 through 7, 9, and 10 will reduce the model from 1597 to 1596 parameters while retaining at least 99% accuracy, because the current design retained 99.97% accuracy with only coordinates 8 and 11 learned, while prior evidence specifically identifies coordinate 8—not coordinate 11—as initialization-sensitive.

INTENDED_EDIT: Retain a learned `fc1` bias only for hidden unit 8 and reconstruct every other bias coordinate as a fixed zero while preserving ordinary linear-layer initialization RNG use.

EVIDENCE: The 1597-parameter design achieved 99.97% accuracy after removing coordinate 10, whereas removing coordinate 8 previously reduced accuracy to 77.57%; this makes coordinate 11 the most informative remaining hidden-bias constraint to test.

<<<<<<< SEARCH
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinates 0 through 7, 9, and 10 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        bias = torch.cat((zeros(8), self.bias_rest[:1], zeros(2), self.bias_rest[1:]))
        return F.linear(x, self.weight, bias)
=======
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with only bias coordinate 8 learned."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 11))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        bias = torch.cat((zeros(8), self.bias_rest, zeros(3)))
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE