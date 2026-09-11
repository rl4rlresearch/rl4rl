MECHANISM: Incremental softmax-invariant key-bias anchoring

HYPOTHESIS: Fixing a third key-bias coordinate at zero will reduce the model from 1,641 to 1,640 parameters while retaining at least 99% accuracy, because each position-independent key-bias coordinate adds only a common shift to a query’s attention logits.

INTENDED_EDIT: Store 21 learned QKV bias coordinates and insert three fixed-zero coordinates at the start of the key-bias segment.

EVIDENCE: Fixing two key-bias coordinates achieved 99.62% accuracy with 1,641 parameters, after one coordinate achieved 99.98%; since removing all key biases at once failed, the smallest incremental extension is the most informative next test.

<<<<<<< SEARCH
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
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with three fixed-zero key-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-3].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.bias[: self.d_model], self.bias.new_zeros(3), self.bias[self.d_model :])
        )
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE