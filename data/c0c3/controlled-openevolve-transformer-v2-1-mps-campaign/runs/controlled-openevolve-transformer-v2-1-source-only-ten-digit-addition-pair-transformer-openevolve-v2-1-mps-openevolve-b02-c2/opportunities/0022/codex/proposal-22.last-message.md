MECHANISM: Cross-projection per-head bias-capacity balancing

HYPOTHESIS: Omitting one value-bias coordinate from head 0 and two from head 1 alongside the qualified 4:3 key-bias layout will produce a 1,634-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reconstruct seven key-bias coordinates and three value-bias coordinates as zeros, distributing total key-plus-value omissions evenly at five per attention head while preserving baseline initialization draws and all QKV weights.

EVIDENCE: The head-balanced value omission design achieved 99.67% at 1,635 parameters, whereas two value omissions concentrated in head 0 achieved only 50.34%; assigning the additional omission to head 1 balances total omissions across the asymmetric 4:3 key layout.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection with key-bias omissions distributed 3:2 across heads."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[key_start + 3 : second_head_start],
                linear.bias[second_head_start + 2 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middle_start = self.key_start
        middle_end = self.second_head_start - 3
        full_bias = torch.cat(
            (
                self.bias[:middle_start],
                self.bias.new_zeros(3),
                self.bias[middle_start:middle_end],
                self.bias.new_zeros(2),
                self.bias[middle_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and three value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[second_head_start + 3 : self.value_start],
                linear.bias[self.value_start + 1 : self.value_start + self.head_dim],
                linear.bias[self.value_start + self.head_dim + 2 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        first_value_end = retained_key_end + self.head_dim - 1
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(1),
                self.bias[retained_key_end:first_value_end],
                self.bias.new_zeros(2),
                self.bias[first_value_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing five
        # redundant key-bias coordinates, distributed 3:2 across the heads.
=======
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3), one value bias from head 0, and two from head 1.
>>>>>>> REPLACE