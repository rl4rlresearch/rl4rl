MECHANISM: Fifth attention-LayerNorm bias absorption

HYPOTHESIS: Fixing `ln1` bias coordinate three at zero will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because its position-independent QKV contribution can be absorbed by the retained query/value biases and is softmax-null for keys.

INTENDED_EDIT: Extend the verified four-coordinate `ln1` bias quotient to coordinate three while leaving all attention weights and query/value biases unchanged.

EVIDENCE: The 1,525-parameter design achieves 99.88% accuracy with coordinates four through seven already removed from `ln1` bias. Unlike the failed value-bias removal, this preserves the full learned value bias and applies the same established `ln1` reparameterization to one additional coordinate.

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