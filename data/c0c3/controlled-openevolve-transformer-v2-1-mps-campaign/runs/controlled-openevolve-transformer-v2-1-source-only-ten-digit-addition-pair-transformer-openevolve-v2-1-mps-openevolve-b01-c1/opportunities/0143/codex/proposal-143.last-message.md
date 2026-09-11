MECHANISM: Complementary attention-input LayerNorm bias quotient

HYPOTHESIS: Fixing `ln1` bias coordinate zero while retaining coordinate one will reduce the model from 1,520 to 1,519 parameters and retain at least 99% accuracy, testing whether the prior collapse was specific to removing coordinate one rather than requiring two learned bias coordinates.

INTENDED_EDIT: Replace the two-coordinate `ln1` bias with only its second coordinate and reconstruct all other coordinates as zeros.

EVIDENCE: Removing coordinates three and two preserved 99.91% and 99.96% accuracy, while removing coordinate one collapsed accuracy to 72.53%; retaining coordinate one and independently removing coordinate zero is the smallest test that distinguishes coordinate-specific importance from a two-parameter capacity threshold.

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
    """LayerNorm retaining only bias coordinate one for attention optimization."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[1:2].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (1, 6))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = SixPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = SevenPrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE