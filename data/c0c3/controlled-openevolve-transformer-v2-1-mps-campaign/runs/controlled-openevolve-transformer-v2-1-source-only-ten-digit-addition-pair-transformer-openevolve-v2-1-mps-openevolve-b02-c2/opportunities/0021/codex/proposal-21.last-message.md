MECHANISM: Head-balanced value-to-output bias reparameterization

HYPOTHESIS: Omitting one value-bias coordinate from each attention head while retaining the qualified 4:3 key-bias layout will produce a 1,635-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend `CompactQKV` to reconstruct the first value-bias coordinate of each head as zero while preserving all initialized QKV weights and the baseline initialization sequence.

EVIDENCE: One value-bias omission achieved 99.92% at 1,636 parameters, while two consecutive omissions in the first head fell to 50.34%; key-bias experiments showed that distributing omissions across heads avoided similar optimization collapse.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection with key-bias omissions distributed 4:3 across heads."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[key_start + 4 : second_head_start],
                linear.bias[second_head_start + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middle_start = self.key_start
        middle_end = self.second_head_start - 4
        full_bias = torch.cat(
            (
                self.bias[:middle_start],
                self.bias.new_zeros(4),
                self.bias[middle_start:middle_end],
                self.bias.new_zeros(3),
                self.bias[middle_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing seven
        # redundant key-bias coordinates, distributed 4:3 across the heads.
=======
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and one redundant value bias from each head.
>>>>>>> REPLACE