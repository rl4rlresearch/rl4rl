MECHANISM: Symmetric three-coordinate key-bias gauge fixing per attention head

HYPOTHESIS: Omitting six redundant key-bias coordinates evenly distributed 3:3 across the two heads will reduce the model to 1,638 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Generalize post-initialization QKV compaction to reconstruct three zero key-bias coordinates in each head while preserving the baseline initialization sequence and every QKV weight.

EVIDENCE: The near-balanced 3:2 omission achieved 99.42% at 1,639 parameters, while balanced omissions consistently outperformed concentrated ones; extending it to a symmetric 3:3 layout is the smallest informative reduction.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection with one softmax-invariant key-bias coordinate omitted."""

    def __init__(self, linear: nn.Linear, omitted_bias: int):
        super().__init__()
        self.omitted_bias = omitted_bias
        self.weight = linear.weight
        compact_bias = torch.cat(
            (linear.bias[:omitted_bias], linear.bias[omitted_bias + 1 :])
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.omitted_bias],
                self.bias.new_zeros(1),
                self.bias[self.omitted_bias :],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Compact only after every module has consumed the baseline initialization
        # draws, so all retained parameters start identically to the qualified run.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(block.attn.qkv, cfg.d_model)
=======
        # Preserve every baseline initialization draw before removing three
        # redundant key-bias coordinates from each attention head.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
>>>>>>> REPLACE