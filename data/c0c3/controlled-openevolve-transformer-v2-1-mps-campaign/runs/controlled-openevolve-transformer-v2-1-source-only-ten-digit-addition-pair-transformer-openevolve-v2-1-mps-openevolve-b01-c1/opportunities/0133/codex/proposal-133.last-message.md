MECHANISM: Fifth attention-input LayerNorm bias quotient

HYPOTHESIS: Fixing `ln1` bias coordinate three at zero will reduce the model from 1,524 to 1,523 parameters while retaining at least 99% accuracy, because its query and value effects can be absorbed by trainable QKV biases while its key effect is softmax-null.

INTENDED_EDIT: Retain only the first three learned `ln1` bias coordinates, reconstruct coordinates three through seven as zeros, and use the renamed five-pruned LayerNorm.

EVIDENCE: The verified 1,524-parameter design already fixes four `ln1` bias coordinates and reaches 99.98% accuracy; removing one additional zero-initialized bias is a smaller extension of that successful quotient than the failed positional-profile and LayerNorm-scale restrictions.

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