MECHANISM: Sixth attention-input bias quotient

HYPOTHESIS: Fixing `ln1` bias coordinate two at zero will reduce the model from 1,523 to 1,522 parameters while retaining at least 99% accuracy, because its constant query and value effects remain representable by learned QKV biases and its constant key effect is softmax-null.

INTENDED_EDIT: Extend the verified `ln1` bias pruning by one adjacent coordinate, retaining only bias coordinates zero and one and reconstructing coordinates two through seven as zeros.

EVIDENCE: Fixing `ln1` bias coordinate three at zero preserved 99.95% accuracy at 1,523 parameters; applying the same quotient to the next adjacent coordinate is the smallest informative reduction while leaving the load-bearing attention widths and LayerNorm scales unchanged.

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