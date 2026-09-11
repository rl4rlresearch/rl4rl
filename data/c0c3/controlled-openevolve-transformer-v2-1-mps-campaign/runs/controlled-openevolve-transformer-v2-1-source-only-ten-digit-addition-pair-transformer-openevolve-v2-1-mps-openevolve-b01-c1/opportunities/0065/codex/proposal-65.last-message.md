MECHANISM: Fifth pre-attention LayerNorm bias quotient

HYPOTHESIS: Removing a fifth `ln1` bias coordinate will reduce the model from 1,552 to 1,551 parameters while retaining at least 99% accuracy, because four consecutive coordinates were pruned successfully and downstream query/value biases can absorb the omitted affine effect while its key-side shift is softmax-invariant.

INTENDED_EDIT: Replace the four-coordinate-pruned LayerNorm before attention with a five-coordinate-pruned variant, leaving initialization, `ln2`, and training unchanged.

EVIDENCE: Extending `ln1` bias pruning from three to four coordinates improved the verified result from 99.91% at 1,553 parameters to 100% at 1,552 parameters, directly motivating one further incremental quotient.

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