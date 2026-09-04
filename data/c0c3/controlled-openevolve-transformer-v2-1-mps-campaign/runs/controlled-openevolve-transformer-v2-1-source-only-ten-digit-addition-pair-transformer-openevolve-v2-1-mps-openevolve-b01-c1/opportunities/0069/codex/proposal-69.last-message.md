MECHANISM: Fifth pre-MLP LayerNorm bias quotient

HYPOTHESIS: Removing a fifth `ln2` bias coordinate will reduce the model from 1,550 to 1,549 parameters while retaining at least 99% accuracy, because the following learned `fc1` bias can absorb its constant affine contribution.

INTENDED_EDIT: Extend pre-MLP LayerNorm bias pruning from four coordinates to five while leaving attention, initialization, and training unchanged.

EVIDENCE: Removing the fourth `ln2` bias coordinate retained 99.98% accuracy at 1,550 parameters; unlike the failed fifth `ln1` pruning, this pathway is immediately followed by the learned `fc1` bias that motivated the successful `ln2` reductions.

<<<<<<< SEARCH
class FourPrunedMLPBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with four bias coordinates absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-4].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class FivePrunedMLPBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with five bias coordinates absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-5].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 5))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = FourPrunedMLPBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = FivePrunedMLPBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE