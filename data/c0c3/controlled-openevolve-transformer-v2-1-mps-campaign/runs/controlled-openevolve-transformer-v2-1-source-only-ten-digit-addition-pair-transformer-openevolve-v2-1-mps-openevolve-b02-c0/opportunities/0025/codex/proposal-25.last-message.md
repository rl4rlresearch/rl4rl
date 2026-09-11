MECHANISM: Dynamic value-bias tying

HYPOTHESIS: Tying the last value-bias coordinate to the first will reduce the model from 1,633 to 1,632 parameters while retaining at least 99% accuracy, because attention-weighted value bias is a learned residual-stream offset that the unrestricted output projection bias can absorb.

INTENDED_EDIT: Store seven value-bias coordinates and reuse the first as the eighth, while preserving the successful query-tied key-bias layout and the initial all-zero function.

EVIDENCE: Dynamic tying succeeded for the first `ln1` scale reduction at 99.78% and for the query/key bias reduction at 99.32%, whereas a second `ln1` scale tie failed; this motivates testing a single learned tie in the distinct, projection-redundant value-bias pathway.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with a query-tied key bias across heads."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-8].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias[self.d_model :],
            )
        )
        return F.linear(x, self.weight, bias)
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key and value bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-9].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = torch.cat(
            (self.bias[self.d_model :], self.bias[self.d_model : self.d_model + 1])
        )
        bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                value_bias,
            )
        )
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE