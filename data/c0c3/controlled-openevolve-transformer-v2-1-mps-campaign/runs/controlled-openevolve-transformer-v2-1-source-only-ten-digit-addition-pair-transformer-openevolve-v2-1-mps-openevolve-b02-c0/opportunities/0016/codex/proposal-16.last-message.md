MECHANISM: Initialization-preserving elimination of the final invariant key-bias scalar

HYPOTHESIS: Replacing the shared key-bias scalar with eight fixed zeros will reduce the model from 1,636 to 1,635 parameters while retaining at least 99% accuracy, because the scalar is initialized to zero and any position-independent key bias cancels under attention softmax.

INTENDED_EDIT: Keep the combined QKV layer and its current initialization path, but store only the eight query and eight value bias coordinates and reconstruct the entire key-bias segment as fixed zeros.

EVIDENCE: The head-shared key-bias model achieved 99.92% with 1,636 parameters; unlike the failed all-at-once removal, this edit starts from the successful staged layout and preserves its registered Linear bias initialization while removing only the final functionally invariant scalar.

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
    """Combined QKV projection with a fixed-zero key bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-8].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(self.d_model),
                self.bias[self.d_model :],
            )
        )
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE