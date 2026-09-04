MECHANISM: First-coordinate alternate LayerNorm-bias quotient

HYPOTHESIS: Removing `ln1` bias coordinate zero instead of the previously failed coordinate three will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because equivalent quotient constraints have shown strong coordinate-specific optimization behavior.

INTENDED_EDIT: Retain trainable `ln1` bias coordinates one through three and reconstruct coordinate zero plus the final four coordinates as zeros; fresh initialization remains unchanged.

EVIDENCE: The 1,534-parameter model reached 99.91% with four bias coordinates removed, but removing coordinate three next collapsed to 11.19%. Earlier MLP experiments showed that changing only the constrained coordinate rescued row 7 from 40.22% to 99.95% and row 5 succeeded only on its second coordinate, motivating an alternate coordinate for the same fifth additive quotient.

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
class AlternateFivePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with five bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[1:-4].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (1, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = FourPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = AlternateFivePrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE