MECHANISM: Head-shared key-bias anchoring

HYPOTHESIS: Sharing the single learned key-bias coordinate across both attention heads will reduce the model from 1,637 to 1,636 parameters while retaining at least 99% accuracy, because the duplicated coordinate remains position-independent and softmax-invariant while preserving the successful head-balanced bias layout.

INTENDED_EDIT: Store one shared key-bias parameter instead of two and insert that same learned scalar as the fourth key-bias coordinate in each head.

EVIDENCE: Six head-balanced key-bias anchors achieved 99.91% accuracy with 1,637 parameters, whereas six consecutive anchors narrowly failed at 98.84%; sharing the two remaining balanced coordinates is the smallest reduction that preserves the successful per-head geometry.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with three fixed key-bias coordinates per head."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-6].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(3),
                self.bias[self.d_model : self.d_model + 1],
                self.bias.new_zeros(3),
                self.bias[self.d_model + 1 : self.d_model + 2],
                self.bias[self.d_model + 2 :],
            )
        )
        return F.linear(x, self.weight, bias)
=======
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
>>>>>>> REPLACE