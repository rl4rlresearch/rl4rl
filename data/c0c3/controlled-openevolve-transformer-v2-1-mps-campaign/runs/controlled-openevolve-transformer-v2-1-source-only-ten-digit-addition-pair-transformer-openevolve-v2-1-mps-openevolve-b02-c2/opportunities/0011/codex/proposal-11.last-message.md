MECHANISM: Symmetric per-head key-bias gauge fixing

HYPOTHESIS: Omitting four redundant key-bias coordinates evenly distributed 2:2 across the two attention heads will produce a 1,640-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the successful head-balanced compact QKV projection by omitting a second key-bias coordinate from head 1 while preserving the complete baseline initialization sequence.

EVIDENCE: The 2:1 head-balanced omission achieved 99.93% at 1,641 parameters, whereas three consecutive omissions achieved only 71.13%; this motivates testing the smallest further reduction while maintaining equal omissions across heads.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection with three head-balanced key-bias coordinates omitted."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[key_start + 2 : second_head_start],
                linear.bias[second_head_start + 1 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middle_start = self.key_start
        middle_end = self.second_head_start - 2
        full_bias = torch.cat(
            (
                self.bias[:middle_start],
                self.bias.new_zeros(2),
                self.bias[middle_start:middle_end],
                self.bias.new_zeros(1),
                self.bias[middle_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """QKV projection with two key-bias coordinates omitted per head."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[key_start + 2 : second_head_start],
                linear.bias[second_head_start + 2 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middle_start = self.key_start
        middle_end = self.second_head_start - 2
        full_bias = torch.cat(
            (
                self.bias[:middle_start],
                self.bias.new_zeros(2),
                self.bias[middle_start:middle_end],
                self.bias.new_zeros(2),
                self.bias[middle_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing redundant
        # key-bias coordinates, distributing the omissions across both heads.
=======
        # Preserve every baseline initialization draw before removing two
        # redundant key-bias coordinates from each attention head.
>>>>>>> REPLACE