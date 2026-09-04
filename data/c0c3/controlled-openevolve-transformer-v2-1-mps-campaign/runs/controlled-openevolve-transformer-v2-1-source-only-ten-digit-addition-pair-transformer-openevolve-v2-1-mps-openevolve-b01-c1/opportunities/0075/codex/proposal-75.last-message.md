MECHANISM: Pre-MLP scale/weight quotient

HYPOTHESIS: Fixing one `ln2` scale coordinate at one will reduce the model from 1,547 to 1,546 parameters while retaining at least 99% accuracy, because the following unconstrained `fc1` weight column can absorb that scale and the initialized function remains unchanged.

INTENDED_EDIT: Store seven learned scale coordinates in the bias-free pre-MLP LayerNorm and reconstruct the eighth as a fixed identity scale.

EVIDENCE: The 1,547-parameter design reached 99.88% after successfully quotienting all `ln2` bias coordinates into `fc1`; this tests the analogous downstream-affine redundancy one coordinate at a time while preserving the full MLP preactivation function family.

<<<<<<< SEARCH
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm whose bias is absorbed completely by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)
=======
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """Bias-free LayerNorm with one scale absorbed by the following linear."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(full_weight[:-1].clone())
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
>>>>>>> REPLACE