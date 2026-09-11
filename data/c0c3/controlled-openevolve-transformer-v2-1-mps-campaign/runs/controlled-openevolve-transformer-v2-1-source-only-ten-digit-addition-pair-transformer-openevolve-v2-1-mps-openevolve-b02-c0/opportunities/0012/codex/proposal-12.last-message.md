MECHANISM: Incremental softmax-invariant key-bias anchoring

HYPOTHESIS: Fixing a fifth key-bias coordinate at zero will reduce the model from 1,639 to 1,638 parameters while retaining at least 99% accuracy, because every position-independent key-bias coordinate contributes only a common shift to each query’s attention logits.

INTENDED_EDIT: Store 19 learned QKV bias coordinates and insert five fixed-zero coordinates at the start of the key-bias segment.

EVIDENCE: Fixing four key-bias coordinates achieved 99.93% accuracy with 1,639 parameters, following successful one-, two-, and three-coordinate anchors; this motivates the smallest incremental extension of the demonstrated redundancy removal.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with four fixed-zero key-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-4].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.bias[: self.d_model], self.bias.new_zeros(4), self.bias[self.d_model :])
        )
        return F.linear(x, self.weight, bias)
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with five fixed-zero key-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-5].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.bias[: self.d_model], self.bias.new_zeros(5), self.bias[self.d_model :])
        )
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE