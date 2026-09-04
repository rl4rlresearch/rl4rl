MECHANISM: Incremental MLP LayerNorm bias absorption

HYPOTHESIS: Fixing a third `ln2` bias coordinate to zero will reduce parameters from 1,615 to 1,614 while retaining at least 99% accuracy, because the omitted coordinate can be absorbed into the retained `fc1` bias.

INTENDED_EDIT: Retain five of eight second pre-LayerNorm bias coordinates and reconstruct the final three as fixed zeros, without changing other model capacity or training settings.

EVIDENCE: Pruning one and then two `ln2` bias coordinates achieved 99.98% and 99.97% accuracy respectively; a third coordinate is the smallest extension of this successful exact bias-absorption mechanism.

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