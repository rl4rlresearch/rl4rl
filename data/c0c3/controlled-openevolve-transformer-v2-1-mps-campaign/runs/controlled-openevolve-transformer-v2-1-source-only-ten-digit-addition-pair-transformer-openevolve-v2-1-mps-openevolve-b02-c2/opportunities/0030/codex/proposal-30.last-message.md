MECHANISM: Common-mode attention residual bias quotient

HYPOTHESIS: Preserving the qualified 4:3 value-bias layout while fixing the attention output bias’s LayerNorm-invisible common-mode degree of freedom will produce a 1,629-parameter model with at least 99% accuracy.

INTENDED_EDIT: Adopt the verified 4:3 per-head value-bias omission and replace the eight-parameter attention projection bias with seven learned relative coordinates, reconstructing the final coordinate as zero.

EVIDENCE: The 4:3 value layout achieved 99.81% at 1,630 parameters, while removing head 1’s final value bias failed. Unlike the failed value/output tie, this patch preserves that scalar and removes only a uniform attention-residual shift annihilated by the following LayerNorms.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and three value biases per head."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[second_head_start + 3 : self.value_start],
                linear.bias[self.value_start + 3 : self.value_start + self.head_dim],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        first_value_end = retained_key_end + self.head_dim - 3
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(3),
                self.bias[retained_key_end:first_value_end],
                self.bias.new_zeros(3),
                self.bias[first_value_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and value biases in a 4:3 layout."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[second_head_start + 3 : self.value_start],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(4),
                self.bias.new_zeros(3),
                self.bias[retained_key_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)


class CompactAttentionProjection(nn.Module):
    """Output projection fixing its LayerNorm-invisible common bias component."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        relative_bias = linear.bias[:-1] - linear.bias[-1]
        self.bias = nn.Parameter(relative_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and three value biases from each head.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Preserve every baseline initialization draw before removing seven
        # key biases, value biases in a 4:3 layout, and the attention output
        # bias's common-mode degree of freedom.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
            block.attn.proj = CompactAttentionProjection(block.attn.proj)
>>>>>>> REPLACE