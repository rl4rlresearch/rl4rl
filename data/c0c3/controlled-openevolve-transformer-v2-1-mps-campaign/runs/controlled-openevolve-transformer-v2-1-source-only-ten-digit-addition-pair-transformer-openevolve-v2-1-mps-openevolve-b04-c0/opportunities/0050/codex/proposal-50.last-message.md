MECHANISM: Fifth hidden-unit bias constraint

HYPOTHESIS: Fixing coordinate 4 of `fc1.bias` at zero will reduce the model from 1603 to 1602 parameters while retaining at least 99% accuracy, because coordinates 0 through 3 were removed consecutively with at least 99.95% accuracy and the hidden units have symmetric initialization.

INTENDED_EDIT: Replace the first MLP projection’s eight learned bias coordinates with seven, reconstructing bias coordinates 0 through 4 as fixed zeros while preserving ordinary linear-layer initialization RNG use.

EVIDENCE: The current 1603-parameter design achieved 99.95% accuracy after fixing `fc1.bias` coordinate 3, continuing three earlier successful consecutive removals; extending this established sequence is the most informative next test.

<<<<<<< SEARCH
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinates 0 through 3 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros(4)
        bias = torch.cat((zeros, self.bias_rest))
        return F.linear(x, self.weight, bias)
=======
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinates 0 through 4 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros(5)
        bias = torch.cat((zeros, self.bias_rest))
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE