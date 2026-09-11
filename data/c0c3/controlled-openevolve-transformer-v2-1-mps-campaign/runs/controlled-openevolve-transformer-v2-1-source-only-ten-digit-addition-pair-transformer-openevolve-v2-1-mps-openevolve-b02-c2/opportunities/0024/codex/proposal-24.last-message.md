MECHANISM: Cross-projection per-head bias-capacity balancing

HYPOTHESIS: Omitting two value-bias coordinates from head 0 and three from head 1 alongside the qualified 4:3 key-bias layout will produce a 1,632-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend `CompactQKV` to reconstruct two head-0 and three head-1 value-bias coordinates as zeros, balancing total key-plus-value omissions at six per head while preserving all QKV weights and baseline initialization draws.

EVIDENCE: The balanced 2:2 value-bias design achieved 99.96% at 1,633 parameters, and the 1:2 layout achieved 99.95%; assigning the next omission to head 1 balances the asymmetric 4:3 key omissions and is the smallest informative reduction.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and one value bias per head."""

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
                linear.bias[self.value_start + self.head_dim + 1 :],
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
                self.bias.new_zeros(1),
                self.bias[first_value_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and value biases in a 2:3 layout."""

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
                linear.bias[self.value_start + 2 : self.value_start + self.head_dim],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        first_value_end = retained_key_end + self.head_dim - 2
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(2),
                self.bias[retained_key_end:first_value_end],
                self.bias.new_zeros(3),
                self.bias[first_value_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and one redundant value bias from each head.
=======
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and value biases in a 2:3 per-head layout.
>>>>>>> REPLACE