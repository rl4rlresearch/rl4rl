MECHANISM: Near-balanced seven-coordinate key-bias gauge fixing

HYPOTHESIS: Omitting seven redundant key-bias coordinates in a 4:3 distribution will reduce the model to 1,637 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Extend post-initialization QKV compaction to omit all four key-bias coordinates from head 0 and three from head 1, preserving baseline initialization draws and every QKV weight.

EVIDENCE: Symmetric 3:3 omission achieved 99.98% at 1,638 parameters, while complete 4:4 omission failed; 4:3 is the smallest informative step between those results.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection with three key-bias coordinates omitted per head."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[key_start + 3 : second_head_start],
                linear.bias[second_head_start + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middle_start = self.key_start
        middle_end = self.second_head_start - 3
        full_bias = torch.cat(
            (
                self.bias[:middle_start],
                self.bias.new_zeros(3),
                self.bias[middle_start:middle_end],
                self.bias.new_zeros(3),
                self.bias[middle_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing three
        # redundant key-bias coordinates from each attention head.
=======
        # Preserve every baseline initialization draw before removing seven
        # redundant key-bias coordinates, distributed 4:3 across the heads.
>>>>>>> REPLACE