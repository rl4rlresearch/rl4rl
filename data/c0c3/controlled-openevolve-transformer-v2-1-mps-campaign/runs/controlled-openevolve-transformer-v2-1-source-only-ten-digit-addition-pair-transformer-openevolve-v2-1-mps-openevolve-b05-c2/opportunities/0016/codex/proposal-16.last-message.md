MECHANISM: Value-tied attention output-bias anchoring

HYPOTHESIS: Reconstructing the third anchored projection-bias coordinate from an existing value-bias scalar will yield a 1579-parameter model with at least 99% accuracy, because it preserves a trainable direct output offset while removing its redundant independent parameter.

INTENDED_EDIT: Retain four independent mean-zero attention projection-bias coordinates, derive the fifth from the aligned learned value-bias coordinate, and keep the two already-qualified coordinates fixed at zero.

EVIDENCE: The 1580-parameter two-anchor design achieved 99.97%, while fixing the third coordinate to zero collapsed accuracy to 74.33%; tying that sensitive coordinate to the full learned value-bias pathway tests whether its trainability—not an independent degree of freedom—is required.

<<<<<<< SEARCH
class AnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with one bias coordinate absorbed by value bias."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 1))
        return F.linear(x, weight, bias)
=======
class ValueTiedMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output bias completed through the learned value-bias path."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-3])

    def forward(self, x: torch.Tensor, tied_bias: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        reduced_bias = torch.cat(
            (self.bias, tied_bias.reshape(1), self.bias.new_zeros(2))
        )
        bias = self.basis @ reduced_bias
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = AnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.proj = ValueTiedMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
=======
        y = self.proj(y, self.qv_bias[1, -3])
>>>>>>> REPLACE