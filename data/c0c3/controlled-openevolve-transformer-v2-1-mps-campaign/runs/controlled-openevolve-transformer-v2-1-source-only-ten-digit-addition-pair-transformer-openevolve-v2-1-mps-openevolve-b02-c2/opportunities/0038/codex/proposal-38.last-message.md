MECHANISM: Head-asymmetric query-bias compaction

HYPOTHESIS: Omitting a second head-0 query-bias coordinate while retaining all three head-1 query-bias coordinates will produce a 1,625-parameter model with at least 99% accuracy.

INTENDED_EDIT: Retain two head-0 and three head-1 query-bias coordinates, preserving all QKV weights, the one-coordinate positional quotient, all key-bias omissions, and the qualified 4:3 value-bias layout.

EVIDENCE: The 1,626-parameter design achieved 99.86%, but removing a second head-1 query bias collapsed accuracy to 72.9%; independently, the successful 4:3 versus failed 3:4 value-bias layouts show that head 0 tolerates stronger bias compaction than head 1.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting one query bias per head, all key biases, and 4:3 value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[: self.head_dim - 1],
                linear.bias[self.head_dim : key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_query_end = self.head_dim - 1
        second_query_end = 2 * first_query_end
        full_bias = torch.cat(
            (
                self.bias[:first_query_end],
                self.bias.new_zeros(1),
                self.bias[first_query_end:second_query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[second_query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """QKV projection retaining 2:3 query biases, no key biases, and 0:1 value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[: self.head_dim - 2],
                linear.bias[self.head_dim : key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_query_end = self.head_dim - 2
        second_query_end = first_query_end + self.head_dim - 1
        full_bias = torch.cat(
            (
                self.bias[:first_query_end],
                self.bias.new_zeros(2),
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
        # query bias per head, all key biases, and 4:3 value biases.
=======
        # Preserve every baseline initialization draw before retaining two
        # head-0 and three head-1 query biases, no key biases, and one value bias.
>>>>>>> REPLACE