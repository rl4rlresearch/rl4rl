MECHANISM: Incremental LayerNorm-to-MLP bias quotient

HYPOTHESIS: Extending the qualified 1,623-parameter design by fixing a second `ln2` bias coordinate at zero will produce a 1,622-parameter model with at least 99% accuracy, because both removed offsets can be absorbed by `fc1.bias`.

INTENDED_EDIT: Reproduce the verified shared head-0 query-bias layout, then retain six learned `ln2` bias coordinates and reconstruct the final two as zero.

EVIDENCE: The 1,623-parameter design fixing one `ln2` bias coordinate achieved 99.67%; this tests the smallest incremental extension of that successful exact affine-offset redundancy without further modifying fragile attention capacity.

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
=======
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
    """LayerNorm with two downstream-linear-absorbed bias coordinates fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-2].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(2)))
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = CompactLayerNormBias(nn.LayerNorm(cfg.d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before retaining two
        # head-0 and three head-1 query biases, no key biases, and one value bias.
=======
        # Retain the qualified 2:3 query and 0:1 value layouts with no key
        # biases, sharing the two retained head-0 query coordinates.
>>>>>>> REPLACE