MECHANISM: Cross-projection per-head bias-capacity balancing

HYPOTHESIS: Omitting one head-1 query-bias coordinate while preserving its critical final value-bias scalar will produce a 1,627-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reconstruct the final head-1 query bias as zero, retaining all QKV weights, all key-bias omissions, the verified positional quotient, and the qualified 4:3 value-bias layout.

EVIDENCE: The current 1,628-parameter design achieved 99.95%; the 4:3 value layout achieved 99.81% while the mirrored 3:4 layout collapsed to 74.45%, motivating preservation of head 1’s sole value bias and testing the smallest reduction in the untouched query-bias family. Assigning it to head 1 balances total per-head QKV bias omissions.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting all key biases and value biases in a 4:3 layout."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[self.key_start :],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """QKV projection omitting one query bias, all key biases, and 4:3 value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[: key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query_end = self.key_start - 1
        full_bias = torch.cat(
            (
                self.bias[:query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing all
        # key biases and value biases in a 4:3 per-head layout.
=======
        # Preserve every baseline initialization draw before removing one
        # head-1 query bias, all key biases, and value biases in a 4:3 layout.
>>>>>>> REPLACE