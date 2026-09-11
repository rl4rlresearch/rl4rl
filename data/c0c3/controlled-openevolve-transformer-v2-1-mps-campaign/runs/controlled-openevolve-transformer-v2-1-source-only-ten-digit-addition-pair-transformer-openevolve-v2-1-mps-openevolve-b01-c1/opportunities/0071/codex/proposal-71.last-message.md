MECHANISM: Complete pre-MLP affine-bias quotient

HYPOTHESIS: Replacing the three remaining `ln2` bias parameters with the restored twelfth `fc1` bias will reduce the model from 1,549 to 1,547 parameters while retaining at least 99% accuracy, because `fc1` can absorb every constant contribution of the preceding LayerNorm bias.

INTENDED_EDIT: Make pre-MLP LayerNorm bias-free and restore a conventional fully biased `fc1`, eliminating two redundant learned coordinates without reducing the attainable MLP preactivation-bias space.

EVIDENCE: Five consecutive `ln2` bias coordinates were removed while retaining 99.98% accuracy at 1,549 parameters; unlike the failed attention-side pruning, this LayerNorm is immediately followed by a learned affine layer whose bias can represent the omitted contribution directly.

<<<<<<< SEARCH
        self.fc1 = OnePrunedOutputBiasLinear(d_model, d_ff)
=======
        self.fc1 = nn.Linear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
class FivePrunedMLPBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with five bias coordinates absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-5].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 5))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm whose bias is absorbed completely by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = FivePrunedMLPBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = BiasFreeMLPLayerNorm(cfg.d_model)
>>>>>>> REPLACE