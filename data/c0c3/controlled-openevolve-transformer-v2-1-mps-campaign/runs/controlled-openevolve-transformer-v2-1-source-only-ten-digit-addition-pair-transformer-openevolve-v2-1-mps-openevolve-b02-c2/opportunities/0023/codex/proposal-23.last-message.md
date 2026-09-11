MECHANISM: Head-balanced value-to-output bias reparameterization

HYPOTHESIS: Omitting two value-bias coordinates from each attention head alongside the qualified 4:3 key-bias layout will produce a 1,633-parameter model with at least 99% accuracy.

INTENDED_EDIT: Remove one additional value-bias coordinate from head 0, yielding a symmetric 2:2 value-bias omission while preserving every QKV weight and the baseline initialization sequence.

EVIDENCE: The current 1:2 value-bias layout achieved 99.95% at 1,634 parameters, and the 1:1 head-balanced layout achieved 99.67%, whereas concentrating two omissions in head 0 collapsed to 50.34%; extending the successful distributed pattern to 2:2 is the smallest informative reduction.

<<<<<<< SEARCH
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
=======
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and two value biases per head."""

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
                linear.bias[self.value_start + self.head_dim + 2 :],
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
                self.bias.new_zeros(2),
                self.bias[first_value_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3), one value bias from head 0, and two from head 1.
=======
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and two value biases from each head.
>>>>>>> REPLACE