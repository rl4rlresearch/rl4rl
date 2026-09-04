MECHANISM: Cross-role query/key bias tying

HYPOTHESIS: Tying the shared key-bias scalar to an existing learned query-bias coordinate will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because the key-bias contribution remains position-independent and softmax-invariant while preserving the successful dynamically learned, head-shared key-bias pathway.

INTENDED_EDIT: Store only the eight query and eight value bias coordinates, and reuse the first learned query-bias scalar as the fourth key-bias coordinate in both attention heads.

EVIDENCE: The learned head-shared key-bias design reached 99.92%, whereas replacing that scalar with fixed zeros collapsed accuracy to 45.19%; tying it to an existing learned coordinate removes one parameter while retaining a varying shared key-bias signal.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with one shared key-bias coordinate across heads."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-7].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[self.d_model : self.d_model + 1]
        bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias[self.d_model + 1 :],
            )
        )
        return F.linear(x, self.weight, bias)
=======
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
>>>>>>> REPLACE