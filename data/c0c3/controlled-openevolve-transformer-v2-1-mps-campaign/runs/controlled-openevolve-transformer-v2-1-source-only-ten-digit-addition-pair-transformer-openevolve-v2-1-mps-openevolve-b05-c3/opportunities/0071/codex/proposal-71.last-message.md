MECHANISM: Three-coordinate pre-attention LayerNorm-scale quotient

HYPOTHESIS: Extending the qualified two-coordinate `ln1` scale quotient to three coordinates will reduce the model from 1,318 to 1,317 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while five adaptive scales remain.

INTENDED_EDIT: Replace the current affine first LayerNorm with a bias-free LayerNorm whose final three scales are fixed at one, leaving attention, optimization, checkpoints, and decoding unchanged.

EVIDENCE: The two-coordinate quotient achieved 99.59% accuracy with 1,318 parameters, while the one-coordinate quotient achieved 99.52%; removing all eight scales failed at 42.51%, motivating the next incremental coordinate rather than a large reduction.

<<<<<<< SEARCH
class FixedRouteValueLinear(nn.Linear):
    """Value projection for content-independent learned attention."""
=======
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with three scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 3))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )


class FixedRouteValueLinear(nn.Linear):
    """Value projection for content-independent learned attention."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
        self.ln1 = GaugeFixedScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE