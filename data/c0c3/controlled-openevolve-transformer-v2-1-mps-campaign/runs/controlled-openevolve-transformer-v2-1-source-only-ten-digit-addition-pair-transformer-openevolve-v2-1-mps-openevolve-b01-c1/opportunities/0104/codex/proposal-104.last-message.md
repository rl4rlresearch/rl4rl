MECHANISM: Second-coordinate LayerNorm-bias quotient

HYPOTHESIS: Fixing `ln1` bias coordinate one at zero will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because this coordinate remains untested and the same coordinate constraint succeeds in two adjacent `fc1` rows.

INTENDED_EDIT: Retain trainable `ln1` bias coordinates zero, two, and three, reconstruct coordinate one and the final four coordinates as zeros, and preserve the unchanged zero-bias initialization.

EVIDENCE: Removing coordinates three and zero produced 11.19% and 56.24% accuracy, demonstrating strong coordinate-specific optimization effects; meanwhile, second-coordinate constraints on `fc1` rows five and six are both present in the 99.91%-accurate 1,534-parameter design.

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
        self.bias = nn.Parameter(
            torch.cat((full_bias[:1], full_bias[2:-4])).clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
                self.bias.new_zeros(4),
            )
        )
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = FourPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = FivePrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE