MECHANISM: Downstream-absorbable LayerNorm beta gauge fixing

HYPOTHESIS: Combining the qualified 4:3 key-bias layout with one fixed `ln2` beta coordinate will produce a 1,636-parameter model with at least 99% accuracy because the unrestricted MLP input bias can absorb that coordinate’s effect.

INTENDED_EDIT: Upgrade QKV compaction from 3:2 to 4:3 and, after baseline initialization, replace `ln2` with an equivalent LayerNorm retaining seven learned bias coordinates and reconstructing the omitted coordinate as zero.

EVIDENCE: The 4:3 QKV design achieved 99.89% at 1,637 parameters. Since positional and final residual-bias gauges failed at 1,636, this tests a distinct affine redundancy directly absorbable by the fully learned `fc1` bias.

<<<<<<< SEARCH
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


class CompactLayerNorm(nn.Module):
    """LayerNorm with one downstream-absorbable bias coordinate omitted."""

    def __init__(self, layer_norm: nn.LayerNorm, omitted: int):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.omitted = omitted
        self.weight = layer_norm.weight
        compact_bias = torch.cat(
            (
                layer_norm.bias[:omitted],
                layer_norm.bias[omitted + 1 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.omitted],
                self.bias.new_zeros(1),
                self.bias[self.omitted :],
            )
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing five
        # redundant key-bias coordinates, distributed 3:2 across the heads.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Preserve every baseline initialization draw before removing seven
        # redundant key-bias coordinates and one absorbable ln2 bias coordinate.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
            block.ln2 = CompactLayerNorm(block.ln2, omitted=0)
>>>>>>> REPLACE