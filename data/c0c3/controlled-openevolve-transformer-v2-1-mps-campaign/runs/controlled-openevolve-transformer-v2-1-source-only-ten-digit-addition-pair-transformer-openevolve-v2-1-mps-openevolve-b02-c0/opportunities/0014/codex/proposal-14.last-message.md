MECHANISM: Head-balanced key-bias anchoring

HYPOTHESIS: Fixing six key-bias coordinates while distributing them evenly across the two attention heads will produce a 1,637-parameter model with at least 99% accuracy, because each key bias remains softmax-invariant and the balanced layout may avoid the optimization sensitivity observed with six consecutive anchors.

INTENDED_EDIT: Store 18 QKV bias parameters and reconstruct each four-dimensional key bias with three fixed-zero coordinates and one learned coordinate.

EVIDENCE: Five consecutive key anchors achieved 99.90% accuracy at 1,638 parameters, while six consecutive anchors narrowly missed at 98.84%; balancing the same six exact-redundancy removals across heads tests whether that miss arose from numerical or optimization sensitivity rather than lost capacity.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE