MECHANISM: Head-asymmetric query-bias compaction

HYPOTHESIS: Retaining one head-0 and three head-1 query-bias coordinates will produce a 1,624-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the qualified 1,625-parameter layout by omitting a third head-0 query bias while preserving all QKV weights, three head-1 query biases, the critical head-1 value bias, all key-bias omissions, and the positional quotient.

EVIDENCE: The 1,625-parameter 2:3 query-bias layout achieved 99.99%, whereas reducing head 1 to two query biases collapsed accuracy to 72.9%; the successful head asymmetry motivates placing the next omission in head 0.

<<<<<<< SEARCH
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
=======
class CompactQKV(nn.Module):
    """QKV projection retaining 1:3 query biases, no key biases, and 0:1 value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[: self.head_dim - 3],
                linear.bias[self.head_dim : key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_query_end = self.head_dim - 3
        second_query_end = first_query_end + self.head_dim - 1
        full_bias = torch.cat(
            (
                self.bias[:first_query_end],
                self.bias.new_zeros(3),
                self.bias[first_query_end:second_query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[second_query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing one
        # head-1 query bias, all key biases, and value biases in a 4:3 layout.
=======
        # Preserve every baseline initialization draw before retaining one
        # head-0 and three head-1 query biases, no key biases, and one value bias.
>>>>>>> REPLACE