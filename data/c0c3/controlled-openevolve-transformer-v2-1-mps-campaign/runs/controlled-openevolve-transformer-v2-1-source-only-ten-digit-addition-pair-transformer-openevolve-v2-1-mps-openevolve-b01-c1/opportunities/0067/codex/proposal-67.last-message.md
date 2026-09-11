MECHANISM: Third pre-MLP LayerNorm bias quotient

HYPOTHESIS: Removing a third `ln2` bias coordinate will reduce the model from 1,552 to 1,551 parameters while retaining at least 99% accuracy, because the current 100%-accurate model already removes two such coordinates and the following `fc1` bias can absorb the additional constant affine effect.

INTENDED_EDIT: Replace the two-coordinate-pruned LayerNorm before the MLP with a three-coordinate-pruned variant, leaving attention, initialization, and training unchanged.

EVIDENCE: The current 1,552-parameter design achieves 100% accuracy while pruning two `ln2` bias coordinates; unlike the failed fifth `ln1` pruning, this tests an unextended downstream-affine quotient with a retained learned `fc1` bias pathway.

<<<<<<< SEARCH
class TwoPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with two bias coordinates absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-2].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = ThreePrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE