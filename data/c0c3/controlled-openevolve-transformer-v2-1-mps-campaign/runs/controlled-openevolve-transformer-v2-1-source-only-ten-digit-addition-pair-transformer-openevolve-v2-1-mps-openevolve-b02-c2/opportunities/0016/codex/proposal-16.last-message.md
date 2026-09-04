MECHANISM: Final residual-bias shift gauge fixing

HYPOTHESIS: Retaining the qualified 4:3 key-bias layout while removing one scalar-shift degree of freedom from the MLP output bias will produce a 1,636-parameter model with at least 99% accuracy.

INTENDED_EDIT: Upgrade QKV compaction from 2:2 to the qualified 4:3 layout, then replace the initialized MLP output projection with an equivalent seven-parameter bias whose omitted coordinate represents the uniform shift eliminated by the final LayerNorm.

EVIDENCE: The 4:3 QKV design achieved 99.89% at 1,637 parameters. The positional-gauge attempt at 1,636 failed, motivating a different one-parameter reduction that acts directly on the final residual stream immediately before LayerNorm.

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


class MLP(nn.Module):
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


class CompactResidualLinear(nn.Module):
    """Linear map with the uniform residual-bias shift gauge fixed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        compact_bias = linear.bias[1:] - linear.bias[:1]
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias.new_zeros(1), self.bias))
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
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
        # redundant key-bias coordinates and fixing one residual-shift gauge.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
            block.mlp.fc2 = CompactResidualLinear(block.mlp.fc2)
>>>>>>> REPLACE