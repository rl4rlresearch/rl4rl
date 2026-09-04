MECHANISM: Fourth pre-attention LayerNorm bias quotient

HYPOTHESIS: Removing a fourth `ln1` bias coordinate will reduce the model from 1,553 to 1,552 parameters while retaining at least 99% accuracy, because the verified three-coordinate pruning shows that downstream query and value biases can absorb these constant affine effects while the key-side effect is softmax-invariant.

INTENDED_EDIT: Replace the three-coordinate-pruned LayerNorm used before attention with a four-coordinate-pruned variant, leaving `ln2`, initialization, and training unchanged.

EVIDENCE: The immediately preceding change removed a third `ln1` bias coordinate and achieved 99.91% accuracy at 1,553 parameters; extending the same successful quotient by one coordinate is the most direct incremental test.

<<<<<<< SEARCH
class ThreePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with three bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-3].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 3))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class FourPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with four bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-4].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = ThreePrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = FourPrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE