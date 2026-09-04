MECHANISM: LayerNorm-1 bias absorption through attention biases

HYPOTHESIS: Fixing LayerNorm-1 bias feature 0 will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because its query effect is representable by `q_bias`, its constant-key effect is softmax-invisible, and its value effect is representable by the attention projection bias.

INTENDED_EDIT: Replace the first block’s LayerNorm with a gauge-fixed variant that reconstructs bias feature 0 as zero while retaining all scale coordinates and the other seven learned bias coordinates.

EVIDENCE: The 1604-parameter design achieved 99.92% accuracy while fixing six LayerNorm-2 bias coordinates, including feature 0. This tests the same single-coordinate bias gauge in LayerNorm-1 without extending the attention-output gauges whose fourth-coordinate trials repeatedly timed out or touching feature 5’s sensitive value-bias pathway.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 1, 2, 3, 4, 6 absorbed downstream."""
=======
class GaugeFixedLayerNormBias(nn.Module):
    """LayerNorm with bias feature 0 absorbed into attention biases."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.fixed_bias_index = 0
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.fixed_bias_index],
                self.bias.new_zeros(1),
                self.bias[self.fixed_bias_index :],
            )
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )


class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 1, 2, 3, 4, 6 absorbed downstream."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = GaugeFixedLayerNormBias(cfg.d_model)
>>>>>>> REPLACE