MECHANISM: Head-1 rank-one query-bias sharing

HYPOTHESIS: Tying the first two retained head-1 query-bias coordinates will reduce the qualified model to 1,623 parameters while maintaining at least 99% accuracy, because both coordinate pathways remain active and the analogous head-0 sharing retained 99.64% accuracy.

INTENDED_EDIT: Replace two independent head-1 query-bias scalars with one shared scalar while preserving the existing shared head-0 biases, third head-1 query bias, final head-1 value bias, key-bias omissions, and positional quotient.

EVIDENCE: The current 1,624-parameter design succeeded by sharing two head-0 query biases, whereas zeroing an additional head-0 bias narrowly failed and reducing head 1 by zeroing a coordinate collapsed accuracy; sharing is therefore the most informative way to reduce head-1 capacity without removing another query pathway.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection sharing 2 head-0 query biases and retaining 3 head-1 biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        second_query_end = 1 + self.head_dim - 1
        full_bias = torch.cat(
            (
                self.bias[:1].expand(self.head_dim - 2),
                self.bias.new_zeros(2),
                self.bias[1:second_query_end],
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
    """QKV projection sharing bias pairs in both heads while retaining five pathways."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : self.head_dim + 1],
                linear.bias[self.head_dim + 2 : key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value_start = 3
        full_bias = torch.cat(
            (
                self.bias[:1].expand(self.head_dim - 2),
                self.bias.new_zeros(2),
                self.bias[1:2].expand(self.head_dim - 2),
                self.bias[2:value_start],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[value_start:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified 2:3 query and 0:1 value layouts with no key
        # biases, sharing the two retained head-0 query coordinates.
=======
        # Retain the qualified query and value pathways with no key biases,
        # sharing the first two retained query coordinates within each head.
>>>>>>> REPLACE