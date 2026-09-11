MECHANISM: Second alternative hidden-unit bias anchoring

HYPOTHESIS: Fixing `fc1.bias[10]` in addition to coordinates 0 through 7 and 9 will reduce the model from 1598 to 1597 parameters while retaining at least 99% accuracy, because fixing coordinate 9 achieved 100% accuracy and showed that performance depends on which initialized hidden-unit bias is removed rather than requiring all four previously remaining biases.

INTENDED_EDIT: Retain learned biases only for hidden units 8 and 11, reconstructing bias coordinates 0 through 7, 9, and 10 as fixed zeros while preserving ordinary linear-layer initialization RNG use.

EVIDENCE: The 1598-parameter design reached 100% accuracy after fixing coordinate 9, whereas fixing coordinate 8 reached only 77.57%; testing coordinate 10 is the most direct next probe of whether another initialization-specific hidden bias is redundant.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE