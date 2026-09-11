MECHANISM: Coupled LayerNorm–MLP affine-offset sharing

HYPOTHESIS: Sharing the penultimate `ln2` bias with the final `fc1` bias will produce a 1,621-parameter model with at least 99% accuracy, because it retains both learned affine pathways while `fc1`’s remaining biases preserve full effective preactivation-offset capacity.

INTENDED_EDIT: Split the compact `ln2` bias into six independent coordinates and one shared scalar, then reuse that scalar as the final `fc1` bias instead of learning a separate parameter.

EVIDENCE: The 1,622-parameter value/projection-offset sharing design reached 99.92%, while fixing a second `ln2` bias reached only 97.73%; applying the successful upstream/downstream sharing pattern avoids deleting the sensitive LayerNorm pathway.

<<<<<<< SEARCH
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


class MLP(nn.Module):
=======
class CompactLayerNormBias(nn.Module):
    """LayerNorm with one fixed bias and one downstream-shared bias."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-2].detach().clone())
        self.shared_bias = nn.Parameter(
            layer_norm.bias[-2:-1].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (self.bias, self.shared_bias, self.bias.new_zeros(1))
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )


class CompactSharedFC1(nn.Module):
    """MLP input projection sharing its final bias with LayerNorm."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = linear.weight
        self.bias = nn.Parameter(linear.bias[:-1].detach().clone())
        self.shared_bias = shared_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.shared_bias))
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        # Retain the qualified QKV layout, then share redundant affine offsets
        # across both the attention and LayerNorm-to-MLP boundaries.
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
            block.mlp.fc1 = CompactSharedFC1(
                block.mlp.fc1,
                block.ln2.shared_bias,
            )
>>>>>>> REPLACE