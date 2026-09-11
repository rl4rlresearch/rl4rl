MECHANISM: Incremental downstream bias absorption

HYPOTHESIS: Fixing a third `ln2` bias coordinate at zero will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because `fc1`’s learned bias exactly absorbs any constant contribution from that LayerNorm coordinate.

INTENDED_EDIT: Add a three-pruned-bias LayerNorm and use it only before the MLP, preserving initialization, attention capacity, and all multiplicative scale parameters.

EVIDENCE: The 1,584-parameter design achieved 99.62%, while fixing an `ln2` scale coordinate reached only 97.98%; this tests the less disruptive additive redundancy already used successfully for two `ln2` bias coordinates.

<<<<<<< SEARCH
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with two bias directions absorbed by token-row means."""
=======
class ThreePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with three bias coordinates absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-3].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 3))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with two bias directions absorbed by token-row means."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = ThreePrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE