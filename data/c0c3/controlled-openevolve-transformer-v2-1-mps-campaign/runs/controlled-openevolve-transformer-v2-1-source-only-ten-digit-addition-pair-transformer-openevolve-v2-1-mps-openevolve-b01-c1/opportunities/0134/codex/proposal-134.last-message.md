MECHANISM: Sixth attention-input LayerNorm bias quotient

HYPOTHESIS: Fixing `ln1` bias coordinate two at zero will reduce the model from 1,523 to 1,522 parameters while retaining at least 99% accuracy, because its query and value effects can be absorbed by trainable QKV biases and its position-common key effect is softmax-null.

INTENDED_EDIT: Retain only the first two learned `ln1` bias coordinates, reconstruct coordinates two through seven as zeros, and rename the specialized LayerNorm accordingly.

EVIDENCE: Fixing `ln1` bias coordinate three at zero preserved 99.91% accuracy at 1,523 parameters; this applies the same single-coordinate, zero-initialized quotient while avoiding the optimization disruption seen from larger simultaneous reductions.

<<<<<<< SEARCH
class FivePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with five bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-5].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 5))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class SixPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with six bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-6].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 6))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = FivePrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = SixPrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE