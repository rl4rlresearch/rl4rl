MECHANISM: Alternating per-head value-bias compaction

HYPOTHESIS: Omitting three value-bias coordinates from each attention head alongside the qualified 4:3 key-bias layout will produce a 1,631-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend `CompactQKV` to reconstruct three value-bias coordinates per head as zeros while preserving every QKV weight and the baseline initialization sequence.

EVIDENCE: The 2:2 value layout achieved 99.96% at 1,633 parameters, and extending the less-compacted head to 2:3 improved accuracy to 99.99% at 1,632; the next alternating reduction is 3:3 and retains one learned value-bias coordinate in each head.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and one value bias."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.value_start = 2 * key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[second_head_start + 3 : self.value_start],
                linear.bias[self.value_start + 1 :],
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
                self.bias.new_zeros(1),
                self.bias[retained_key_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and three value biases per head."""

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
                linear.bias[self.value_start + 3 : self.value_start + self.head_dim],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        first_value_end = retained_key_end + self.head_dim - 3
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(3),
                self.bias[retained_key_end:first_value_end],
                self.bias.new_zeros(3),
                self.bias[first_value_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing seven
        # key-bias coordinates (4:3) and one redundant value-bias coordinate.
=======
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and three value biases from each head.
>>>>>>> REPLACE