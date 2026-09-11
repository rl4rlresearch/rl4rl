MECHANISM: Eighth hidden-unit bias constraint

HYPOTHESIS: Fixing coordinate 7 of `fc1.bias` at zero will reduce the model from 1600 to 1599 parameters while retaining at least 99% accuracy, because coordinates 0 through 6 were removed consecutively without violating the accuracy requirement and the hidden units are initialization-symmetric.

INTENDED_EDIT: Replace the first MLP projection’s five learned bias coordinates with four, reconstructing bias coordinates 0 through 7 as fixed zeros while preserving ordinary linear-layer initialization RNG use.

EVIDENCE: The current 1600-parameter design achieved 99.77% accuracy after fixing `fc1.bias` coordinate 6, extending six earlier consecutive successful removals; continuing this established sequence is the most informative next capacity test.

<<<<<<< SEARCH
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinates 0 through 6 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros(7)
        bias = torch.cat((zeros, self.bias_rest))
        return F.linear(x, self.weight, bias)
=======
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
>>>>>>> REPLACE