MECHANISM: Ninth hidden-unit bias constraint

HYPOTHESIS: Fixing coordinate 8 of `fc1.bias` at zero will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because coordinates 0 through 7 were removed consecutively without violating the accuracy requirement and the hidden units are initialization-symmetric.

INTENDED_EDIT: Replace the first MLP projection’s four learned bias coordinates with three, reconstructing bias coordinates 0 through 8 as fixed zeros while preserving ordinary linear-layer initialization RNG use.

EVIDENCE: Fixing `fc1.bias` coordinate 7 achieved 99.91% accuracy at 1599 parameters, extending seven earlier consecutive successful removals; continuing this established sequence is the most informative next capacity test.

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
    """Linear layer with bias coordinates 0 through 8 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros(9)
        bias = torch.cat((zeros, self.bias_rest))
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE