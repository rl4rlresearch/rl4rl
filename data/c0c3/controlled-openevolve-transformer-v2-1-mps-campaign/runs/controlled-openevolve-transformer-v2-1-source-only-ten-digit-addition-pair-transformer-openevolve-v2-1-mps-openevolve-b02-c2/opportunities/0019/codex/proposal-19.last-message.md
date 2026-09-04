MECHANISM: Value-to-output bias reparameterization

HYPOTHESIS: Retaining the qualified 4:3 key-bias layout while omitting one redundant value-bias coordinate will produce a 1,636-parameter model with at least 99% accuracy.

INTENDED_EDIT: Upgrade QKV compaction from 2:2 to 4:3 key-bias omission and reconstruct the first value-bias coordinate as zero; its position-independent effect remains representable by the learned attention output-projection bias.

EVIDENCE: The 4:3 key-bias design achieved 99.89% at 1,637 parameters. Prior 1,636 attempts modified other gauges unsuccessfully, motivating the smallest reduction along the distinct exact redundancy between value bias and output-projection bias.

<<<<<<< SEARCH
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
=======
class CompactQKV(nn.Module):
    """QKV projection omitting seven key biases and one value bias."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.second_head_start = second_head_start
        self.value_start = 2 * key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:key_start],
                linear.bias[second_head_start + 3 : self.value_start],
                linear.bias[self.value_start + 1 :],
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
                self.bias.new_zeros(1),
                self.bias[retained_key_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing two
        # redundant key-bias coordinates from each attention head.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Preserve every baseline initialization draw before removing seven
        # key-bias coordinates (4:3) and one redundant value-bias coordinate.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
>>>>>>> REPLACE