MECHANISM: Fourth orthonormal LayerNorm–key-weight gauge quotient

HYPOTHESIS: Extending the verified three-row key quotient to the second key coordinate of head 1 will yield 1,617 parameters and at least 99% accuracy because the removed direction contributes only a position-independent key shift canceled by causal attention softmax.

INTENDED_EDIT: Reproduce the qualified 1,618-parameter design and reconstruct four QKV key rows from independent seven-dimensional Helmert coordinates scaled by `ln1.weight`.

EVIDENCE: The identical quotient succeeded for three key rows at 1,618 parameters with 99.96% accuracy; applying it to the symmetric second coordinate of the other head is the smallest directly supported reduction.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting one query bias, all key biases, and 4:3 value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[: key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query_end = self.key_start - 1
        full_bias = torch.cat(
            (
                self.bias[:query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """Compact QKV with four LayerNorm-induced key-weight gauges fixed."""

    def __init__(
        self,
        linear: nn.Linear,
        key_start: int,
        second_head_start: int,
        ln_weight: nn.Parameter,
    ):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.second_key_row = second_head_start
        self.ln_weight = ln_weight

        retained_weight = torch.cat(
            (
                linear.weight[:key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 2 :],
            ),
            dim=0,
        )
        self.weight = nn.Parameter(retained_weight.detach().clone())

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("key_basis", basis, persistent=False)

        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    second_head_start,
                    second_head_start + 1,
                ]
            ]
            * ln_weight
        )
        centered_key_weight = (
            scaled_key_weight - scaled_key_weight.mean(dim=1, keepdim=True)
        )
        self.key_weight = nn.Parameter(
            (centered_key_weight @ basis).detach().clone()
        )

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
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        second_retained_start = self.second_key_row - 2
        full_weight = torch.cat(
            (
                self.weight[: self.key_start],
                key_weight[:2],
                self.weight[self.key_start : second_retained_start],
                key_weight[2:],
                self.weight[second_retained_start:],
            ),
            dim=0,
        )
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
        return F.linear(x, full_weight, full_bias)


class CompactSharedProjection(nn.Module):
    """Projection with a zero-mean effective offset and retained value scalar."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = linear.weight
        self.shared_bias = shared_bias

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 2)
        for column in range(width - 2):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_bias = linear.bias - linear.bias.mean()
        compact_bias = basis.transpose(0, 1) @ centered_bias
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_bias = self.bias_basis @ self.bias
        raw_bias = torch.cat(
            (
                raw_bias[:-1],
                raw_bias[-1:] + self.shared_bias,
            )
        )
        value_offset = self.weight[:, -1] * self.shared_bias
        full_bias = raw_bias - (raw_bias + value_offset).mean()
        return F.linear(x, self.weight, full_bias)


class CompactLayerNormBias(nn.Module):
    """LayerNorm with one downstream-linear-absorbed bias coordinate fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-1].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
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
        # Preserve every baseline initialization draw before removing one
        # head-1 query bias, all key biases, and value biases in a 4:3 layout.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Retain the qualified compact bias and projection-offset layout while
        # fixing two independent key-weight gauges in each attention head.
        for block in self.blocks:
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
                block.ln1.weight,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
>>>>>>> REPLACE