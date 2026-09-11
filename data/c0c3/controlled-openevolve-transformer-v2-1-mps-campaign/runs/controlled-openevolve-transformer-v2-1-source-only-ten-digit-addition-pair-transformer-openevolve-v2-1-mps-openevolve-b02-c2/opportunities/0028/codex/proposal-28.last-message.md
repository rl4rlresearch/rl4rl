MECHANISM: Attention-local value/output bias gauge tying

HYPOTHESIS: Tying one attention output-bias coordinate to the sole retained head-1 value-bias scalar in the qualified 4:3 layout will produce a 1,629-parameter model with at least 99% accuracy.

INTENDED_EDIT: Adopt the qualified 4:3 value-bias omission layout, then replace the eight-scalar attention output bias with seven independent scalars plus the retained value-bias scalar.

EVIDENCE: The 4:3 layout achieved 99.81% at 1,630 parameters, while 3:4 fell to 74.45%, motivating preservation of head 1’s final value-bias scalar and removal of a redundant degree of freedom from the position-independent output bias instead.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and two value biases per head."""

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
                linear.bias[self.value_start + 2 : self.value_start + self.head_dim],
                linear.bias[self.value_start + self.head_dim + 2 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retained_key_end = self.key_start + 1
        first_value_end = retained_key_end + self.head_dim - 2
        full_bias = torch.cat(
            (
                self.bias[: self.key_start],
                self.bias.new_zeros(7),
                self.bias[self.key_start : retained_key_end],
                self.bias.new_zeros(2),
                self.bias[retained_key_end:first_value_end],
                self.bias.new_zeros(2),
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


class CompactOutputProjection(nn.Module):
    """Output projection tying its first bias coordinate to a value-bias scalar."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = linear.weight
        self.shared_bias = shared_bias
        self.bias = nn.Parameter(linear.bias[1:].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.shared_bias[-1:], self.bias))
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and two value biases from each head.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Preserve every baseline initialization draw before removing seven
        # key biases and value biases in a 4:3 layout, then tie one output-bias
        # coordinate to the sole retained value-bias scalar.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
            block.attn.proj = CompactOutputProjection(
                block.attn.proj,
                block.attn.qkv.bias,
            )
>>>>>>> REPLACE