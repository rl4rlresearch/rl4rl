MECHANISM: Fifth additive LayerNorm-bias quotient

HYPOTHESIS: Fixing a fifth `ln1` bias coordinate at zero will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because its query and value effects are absorbable by learned Q/V biases and its key effect is softmax-null.

INTENDED_EDIT: Retain only the first three trainable `ln1` bias coordinates and reconstruct the remaining five as zeros.

EVIDENCE: The current 1,534-parameter model achieves 99.91% accuracy with four `ln1` bias coordinates already removed. Extending that same additive quotient by one coordinate is the smallest targeted reduction and avoids the multiplicative optimization failure observed when fixing an `ln2` scale.

<<<<<<< SEARCH
class FourPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with four bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-4].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class FivePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with five bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-5].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 5))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = FourPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = FivePrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE