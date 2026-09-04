MECHANISM: Near-balanced per-head key-bias gauge fixing

HYPOTHESIS: Omitting five redundant key-bias coordinates distributed 3:2 across the two attention heads will produce a 1,639-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend post-initialization QKV compaction to omit three key-bias coordinates from head 0 and two from head 1 while preserving all baseline initialization draws.

EVIDENCE: The balanced 2:2 omission achieved 99.71% at 1,640 parameters, while 3:0 failed and 2:1 succeeded; this indicates per-head balance matters and motivates the smallest further reduction, 3:2.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection with two softmax-invariant key-bias coordinates omitted."""

    def __init__(self, linear: nn.Linear, omitted_bias: int):
        super().__init__()
        self.omitted_bias = omitted_bias
        self.weight = linear.weight
        compact_bias = torch.cat(
            (linear.bias[:omitted_bias], linear.bias[omitted_bias + 2 :])
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.omitted_bias],
                self.bias.new_zeros(2),
                self.bias[self.omitted_bias :],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """QKV projection with key-bias omissions distributed 3:2 across heads."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[key_start + 3 : second_head_start],
                linear.bias[second_head_start + 2 :],
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
                self.bias.new_zeros(2),
                self.bias[middle_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the qualified baseline initialization trajectory before
        # removing two functionally redundant key-bias coordinates.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(block.attn.qkv, cfg.d_model)
=======
        # Preserve every baseline initialization draw before removing five
        # redundant key-bias coordinates, distributed 3:2 across the heads.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
>>>>>>> REPLACE