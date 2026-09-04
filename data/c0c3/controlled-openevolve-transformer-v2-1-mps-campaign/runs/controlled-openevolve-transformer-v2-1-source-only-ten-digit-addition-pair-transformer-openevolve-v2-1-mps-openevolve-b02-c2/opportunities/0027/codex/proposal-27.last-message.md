MECHANISM: Head-asymmetric value-bias compaction

HYPOTHESIS: Omitting all four value-bias coordinates from head 0 while retaining one in head 1 will produce a 1,630-parameter model with at least 99% accuracy.

INTENDED_EDIT: Change the current 2:3 value-bias omission layout to 4:3, preserving all QKV weights, the qualified 4:3 key-bias layout, and baseline initialization draws.

EVIDENCE: The 3:3 layout achieved 99.6%, while 3:4 fell to 74.45%; testing 4:3 isolates whether the failure was specifically caused by removing head 1’s final value-bias coordinate.

<<<<<<< SEARCH
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
=======
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and value biases in a 4:3 layout."""

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
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(4),
                self.bias.new_zeros(3),
                self.bias[retained_key_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and value biases in a 2:3 per-head layout.
=======
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and value biases in a 4:3 per-head layout.
>>>>>>> REPLACE