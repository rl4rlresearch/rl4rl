MECHANISM: Third hidden-unit bias constraint

HYPOTHESIS: Fixing coordinate 2 of `fc1.bias` at zero will reduce the model from 1605 to 1604 parameters while retaining at least 99% accuracy, because coordinates 0 and 1 were removed independently with 99.96% and 99.95% accuracy, and the identically initialized hidden units remain symmetric.

INTENDED_EDIT: Replace the first MLP projection’s ten learned bias coordinates with nine, reconstructing bias coordinates 0 through 2 as fixed zeros while preserving ordinary linear-layer initialization RNG use.

EVIDENCE: Consecutively fixing `fc1.bias` coordinates 0 and 1 retained 99.96% and 99.95% accuracy; after the distinct shared-embedding reduction failed at 71.89%, extending the demonstrated MLP-bias trend is the most direct test of whether its practical redundancy continues beyond the two exact LayerNorm-shift degrees of freedom.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE