MECHANISM: Fourth hidden-unit bias constraint

HYPOTHESIS: Fixing coordinate 3 of `fc1.bias` at zero will reduce the model from 1604 to 1603 parameters while retaining at least 99% accuracy, because coordinates 0 through 2 were removed consecutively with at least 99.95% accuracy and all hidden units use symmetric initialization.

INTENDED_EDIT: Replace the first MLP projection’s nine learned bias coordinates with eight, reconstructing bias coordinates 0 through 3 as fixed zeros while preserving ordinary linear-layer initialization RNG use.

EVIDENCE: The current design achieved 99.98% accuracy after fixing `fc1.bias` coordinate 2, extending two earlier successful removals; testing coordinate 3 is the most direct probe of whether this demonstrated hidden-bias redundancy continues.

<<<<<<< SEARCH
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinates 0 through 2 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros(3)
        bias = torch.cat((zeros, self.bias_rest))
        return F.linear(x, self.weight, bias)
=======
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
>>>>>>> REPLACE