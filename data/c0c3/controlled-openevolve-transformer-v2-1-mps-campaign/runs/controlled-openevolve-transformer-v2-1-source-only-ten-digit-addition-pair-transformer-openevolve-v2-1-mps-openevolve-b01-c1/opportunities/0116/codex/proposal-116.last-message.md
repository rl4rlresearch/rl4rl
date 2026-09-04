MECHANISM: Downstream-absorbed final LayerNorm scale

HYPOTHESIS: Fixing `ln2` scale coordinate seven at its initialized value of one will reduce the model from 1,531 to 1,530 parameters while retaining at least 99% accuracy, because its effect can be absorbed by the corresponding trainable `fc1` coefficients.

INTENDED_EDIT: Retain seven learned `ln2` scale coordinates and reconstruct the final coordinate as one during the forward pass.

EVIDENCE: The 1,531-parameter design reached 99.94% accuracy while `fc1` already omits its final input coordinate in seven of twelve rows; the scale is unused in those rows and multiplicatively redundant with the five remaining trainable final-coordinate weights.

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
    """LayerNorm with absorbed bias and final scale fixed at one."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(full_weight[:-1].clone())
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
>>>>>>> REPLACE