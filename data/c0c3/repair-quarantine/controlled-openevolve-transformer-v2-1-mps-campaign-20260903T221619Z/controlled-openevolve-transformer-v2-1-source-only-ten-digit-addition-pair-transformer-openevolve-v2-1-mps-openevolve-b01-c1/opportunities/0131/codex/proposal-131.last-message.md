MECHANISM: Fifth attention-input bias quotient

HYPOTHESIS: Fixing `ln1` bias coordinate three at zero will reduce the model from 1,524 to 1,523 parameters while retaining at least 99% accuracy, because its query and value effects remain representable by the learned QKV biases while its key effect is softmax-null.

INTENDED_EDIT: Extend the verified four-coordinate `ln1` bias pruning to coordinate three, retaining only the first three learned bias coordinates.

EVIDENCE: The 1,524-parameter design achieves 99.99% accuracy with `ln1` bias coordinates four through seven already fixed at zero; this tests one adjacent coordinate using the same quotient while leaving the multiplicative scales and head widths implicated by prior failures unchanged.

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