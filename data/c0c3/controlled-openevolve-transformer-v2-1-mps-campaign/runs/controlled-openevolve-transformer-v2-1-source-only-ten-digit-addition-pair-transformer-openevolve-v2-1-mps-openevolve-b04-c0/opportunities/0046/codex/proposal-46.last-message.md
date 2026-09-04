MECHANISM: Second reverse pre-MLP shift-to-bias gauge fixing

HYPOTHESIS: Fixing coordinate 1 of `fc1.bias` at zero will reduce the model from 1606 to 1605 parameters while retaining at least 99% accuracy, because the second remaining learned pre-MLP LayerNorm shift provides another independent degree of freedom for absorbing a hidden-unit bias constraint.

INTENDED_EDIT: Replace the first MLP projection’s eleven learned bias coordinates with ten, reconstructing bias coordinates 0 and 1 as fixed zeros while preserving ordinary linear-layer initialization RNG use.

EVIDENCE: Fixing `fc1.bias` coordinate 0 achieved 99.96% accuracy at 1606 parameters; hidden units are initialization-symmetric, and two pre-MLP LayerNorm shifts remain learned, motivating the corresponding second constraint.

<<<<<<< SEARCH
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinate 0 absorbed by its input shift."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        bias = torch.cat((zero, self.bias_rest))
        return F.linear(x, self.weight, bias)
=======
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinates 0 and 1 absorbed by input shifts."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros(2)
        bias = torch.cat((zeros, self.bias_rest))
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE