MECHANISM: Shared adjacent LayerNorm-offset quotient

HYPOTHESIS: Building on the verified 1,622-parameter design, tying the penultimate two `ln2` bias coordinates while retaining both pathways will yield 1,621 parameters and at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified shared head-0 query bias, shared value/projection offset, and final-zero `ln2` bias, then share the two adjacent retained `ln2` bias coordinates instead of fixing the second one to zero.

EVIDENCE: The 1,622-parameter value/projection-sharing design achieved 99.92%; independently, sharing two head-0 query biases succeeded at 99.64% where deleting the second pathway missed at 98.68%, motivating sharing rather than the failed second `ln2`-bias deletion.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection retaining 2:3 query biases, no key biases, and 0:1 value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[: self.head_dim - 2],
                linear.bias[self.head_dim : key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_query_end = self.head_dim - 2
        second_query_end = first_query_end + self.head_dim - 1
        full_bias = torch.cat(
            (
                self.bias[:first_query_end],
                self.bias.new_zeros(2),
                self.bias[first_query_end:second_query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[second_query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
=======
class CompactQKV(nn.Module):
    """QKV projection with shared head-0 queries and one retained value bias."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        query_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
            )
        )
        self.query_bias = nn.Parameter(query_bias.detach().clone())
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.query_bias[:1].expand(self.head_dim - 2),
                self.query_bias.new_zeros(2),
                self.query_bias[1:],
                self.query_bias.new_zeros(1),
                self.query_bias.new_zeros(self.key_start),
                self.query_bias.new_zeros(self.head_dim),
                self.query_bias.new_zeros(3),
                self.value_bias,
            )
        )
        return F.linear(x, self.weight, full_bias)


class CompactSharedProjection(nn.Module):
    """Attention projection sharing its final bias with the retained value bias."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = linear.weight
        self.bias = nn.Parameter(linear.bias[:-1].detach().clone())
        self.shared_bias = shared_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.shared_bias))
        return F.linear(x, self.weight, full_bias)


class CompactLayerNormBias(nn.Module):
    """LayerNorm with one fixed and two shared downstream-redundant offsets."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-2].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias,
                self.bias[-1:],
                self.bias.new_zeros(1),
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
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = CompactLayerNormBias(nn.LayerNorm(cfg.d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before retaining two
        # head-0 and three head-1 query biases, no key biases, and one value bias.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Retain the qualified QKV layout, share its value/projection offset,
        # and preserve both adjacent LayerNorm-bias pathways with one scalar.
        for block in self.blocks:
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
>>>>>>> REPLACE