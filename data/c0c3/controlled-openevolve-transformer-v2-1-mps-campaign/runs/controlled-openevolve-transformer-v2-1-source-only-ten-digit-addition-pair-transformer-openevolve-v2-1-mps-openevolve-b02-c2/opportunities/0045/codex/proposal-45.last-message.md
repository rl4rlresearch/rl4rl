MECHANISM: LayerNorm-to-MLP affine-offset quotient

HYPOTHESIS: Fixing one `ln2` bias coordinate at zero will produce a 1,623-parameter model with at least 99% accuracy because `fc1.bias` can absorb its entire downstream effect while the qualified attention layout and all LayerNorm scales remain unchanged.

INTENDED_EDIT: Replace each block’s `ln2` with a compact LayerNorm retaining all eight scales and seven bias coordinates, reconstructing the final bias coordinate as zero.

EVIDENCE: The current 1,624-parameter design achieved 99.64%. Tested 1,623-parameter attention changes and the `ln2` scale quotient failed, motivating a distinct exact affine-offset redundancy that does not alter attention or constrain `ln2` scaling.

<<<<<<< SEARCH
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
=======
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


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = CompactLayerNormBias(nn.LayerNorm(cfg.d_model))
>>>>>>> REPLACE