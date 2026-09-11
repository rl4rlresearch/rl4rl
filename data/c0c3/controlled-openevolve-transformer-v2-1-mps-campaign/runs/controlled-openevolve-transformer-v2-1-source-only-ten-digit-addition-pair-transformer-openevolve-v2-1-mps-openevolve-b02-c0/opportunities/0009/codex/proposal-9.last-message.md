MECHANISM: Incremental softmax-invariant key-bias anchoring

HYPOTHESIS: Fixing a second key-bias coordinate at zero will reduce the model from 1,642 to 1,641 parameters while retaining at least 99% accuracy, because every position-independent key-bias coordinate contributes only a common shift to each query’s attention logits.

INTENDED_EDIT: Store 22 learned QKV bias coordinates and insert two fixed-zero coordinates at the start of the key-bias segment.

EVIDENCE: One key-bias anchor achieved 99.98% accuracy with 1,642 parameters, while removing all key biases at once failed; this motivates the smallest incremental extension of the successful exact redundancy removal.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with one fixed-zero key-bias coordinate."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.bias[: self.d_model], self.bias.new_zeros(1), self.bias[self.d_model :])
        )
        return F.linear(x, self.weight, bias)
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with two fixed-zero key-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-2].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.bias[: self.d_model], self.bias.new_zeros(2), self.bias[self.d_model :])
        )
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE