MECHANISM: Seventh attention-input LayerNorm bias quotient

HYPOTHESIS: Fixing `ln1` bias coordinate one at zero will reduce the model from 1,522 to 1,521 parameters while retaining at least 99% accuracy, because its query and value effects can be absorbed by trainable QKV biases and its position-common key effect is softmax-null.

INTENDED_EDIT: Retain only the first learned `ln1` bias coordinate, reconstruct coordinates one through seven as zeros, and rename the specialized LayerNorm accordingly.

EVIDENCE: Successively fixing `ln1` bias coordinates three and two preserved 99.91% and 99.96% accuracy; this applies the same one-coordinate quotient while minimizing optimization disruption.

<<<<<<< SEARCH
class SixPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with six bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-6].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 6))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class SevenPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with seven bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-7].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 7))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = SixPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = SevenPrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE