MECHANISM: Sixth hidden-unit bias constraint

HYPOTHESIS: Fixing coordinate 5 of `fc1.bias` at zero will reduce the model from 1602 to 1601 parameters while retaining at least 99% accuracy, because coordinates 0 through 4 were removed consecutively without violating the accuracy requirement and the hidden units are initialization-symmetric.

INTENDED_EDIT: Replace the first MLP projection’s seven learned bias coordinates with six, reconstructing bias coordinates 0 through 5 as fixed zeros while preserving ordinary linear-layer initialization RNG use.

EVIDENCE: The current design achieved 99.87% accuracy after fixing `fc1.bias` coordinate 4, following four consecutive successful removals; extending this established sequence is the most direct test of the remaining hidden-bias capacity.

<<<<<<< SEARCH
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
=======
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinates 0 through 5 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros(6)
        bias = torch.cat((zeros, self.bias_rest))
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE