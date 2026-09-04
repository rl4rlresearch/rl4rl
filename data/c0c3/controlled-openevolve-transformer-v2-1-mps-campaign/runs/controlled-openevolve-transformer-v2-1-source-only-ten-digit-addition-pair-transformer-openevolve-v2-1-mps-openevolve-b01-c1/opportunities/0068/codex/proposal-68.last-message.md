MECHANISM: Fourth pre-MLP LayerNorm bias quotient

HYPOTHESIS: Removing a fourth `ln2` bias coordinate will reduce the model from 1,551 to 1,550 parameters while retaining at least 99% accuracy, because the following learned `fc1` bias can absorb the omitted constant affine effect.

INTENDED_EDIT: Extend pre-MLP LayerNorm bias pruning from three coordinates to four, leaving attention, initialization, and training unchanged.

EVIDENCE: Removing the third `ln2` bias coordinate achieved 99.94% accuracy at 1,551 parameters, making one further incremental extension of this still-successful quotient the most informative next test.

<<<<<<< SEARCH
class ThreePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with three bias coordinates absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-3].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 3))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class FourPrunedMLPBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with four bias coordinates absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-4].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = ThreePrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = FourPrunedMLPBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE