MECHANISM: Shared value–projection offset quotient

HYPOTHESIS: Tying the retained final head-1 value bias to the corresponding final attention-projection bias will produce a 1,622-parameter model with at least 99% accuracy while preserving both learned bias pathways.

INTENDED_EDIT: Split the compact QKV bias into query and value parameters, then reuse the value scalar as the final projection-bias coordinate instead of learning a separate scalar.

EVIDENCE: The qualified 1,623-parameter model achieved 99.67% while retaining the critical final head-1 value bias; prior 1,622 designs fixed sensitive coordinates to zero, whereas this sharing removes a redundant constant offset without deleting either pathway.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection sharing 2 head-0 query biases and retaining 3 head-1 biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        second_query_end = 1 + self.head_dim - 1
        full_bias = torch.cat(
            (
                self.bias[:1].expand(self.head_dim - 2),
                self.bias.new_zeros(2),
                self.bias[1:second_query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[second_query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)


class CompactLayerNormBias(nn.Module):
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified 2:3 query and 0:1 value layouts with no key
        # biases, sharing the two retained head-0 query coordinates.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Retain the qualified QKV layout, then quotient its constant-offset
        # redundancy by sharing the final value and projection bias scalar.
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