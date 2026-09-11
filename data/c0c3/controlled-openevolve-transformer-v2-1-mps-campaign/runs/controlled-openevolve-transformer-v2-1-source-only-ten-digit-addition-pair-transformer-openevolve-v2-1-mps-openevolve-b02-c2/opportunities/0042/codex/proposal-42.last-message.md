MECHANISM: Gauge-aware head-1 query-bias sharing

HYPOTHESIS: Sharing the second and third retained head-1 query-bias coordinates, while keeping its first coordinate independent, will produce a 1,623-parameter model with at least 99% accuracy.

INTENDED_EDIT: Apply the qualified head-0 query-bias sharing and additionally tie the two head-1 query coordinates that do not align with the position-embedding gauge-fixed hidden coordinate.

EVIDENCE: Head-0 sharing achieved 99.64% at 1,624 parameters, while tying the first two head-1 coordinates fell to 92.98%; testing the alternative head-1 pair preserves the first coordinate independently and isolates whether that coordinate-specific coupling caused the failure.

<<<<<<< SEARCH
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
=======
class CompactQKV(nn.Module):
    """QKV projection sharing head-0 biases and the final two retained head-1 biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : self.head_dim + 2],
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
                self.bias[1:2],
                self.bias[2:3].expand(2),
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
        # Preserve every baseline initialization draw before retaining two
        # head-0 and three head-1 query biases, no key biases, and one value bias.
=======
        # Preserve every baseline initialization draw while sharing the two
        # head-0 biases and the final two retained head-1 query biases.
>>>>>>> REPLACE