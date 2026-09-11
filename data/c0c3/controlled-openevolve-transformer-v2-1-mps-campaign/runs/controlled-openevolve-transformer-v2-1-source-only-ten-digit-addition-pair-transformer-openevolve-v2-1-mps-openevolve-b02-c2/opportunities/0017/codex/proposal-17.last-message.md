MECHANISM: Parameter-free adaptive key-bias gauge coupling

HYPOTHESIS: Tying the final redundant key-bias coordinate to an existing learned query-bias scalar will reduce the model to 1,636 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove the remaining independent key-bias parameter from the qualified 4:3 design and reconstruct it from the first learned query-bias coordinate.

EVIDENCE: The independent 4:3 key-bias layout achieved 99.89% at 1,637 parameters, whereas fixing all eight key-bias coordinates to zero collapsed to 30.16%; coupling the last coordinate preserves an adaptive numerical offset without adding a parameter.

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
    """QKV projection whose final key-bias coordinate shares a query bias."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[second_head_start + 4 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[:1],
                self.bias[self.key_start :],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE