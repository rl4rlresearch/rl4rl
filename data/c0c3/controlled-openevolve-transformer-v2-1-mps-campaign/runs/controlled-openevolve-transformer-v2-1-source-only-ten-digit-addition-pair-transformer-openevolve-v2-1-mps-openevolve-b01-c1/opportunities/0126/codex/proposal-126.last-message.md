MECHANISM: Downstream-absorbed attention LayerNorm scale

HYPOTHESIS: Fixing `ln1` scale coordinate seven at one will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because all query, key, and value rows have trainable coefficients that can absorb its scale.

INTENDED_EDIT: Retain seven learned `ln1` scales and reconstruct the final scale as one during the forward pass.

EVIDENCE: Fixing `ln2` coordinate seven preserved 99.96% accuracy despite sparse downstream use; `ln1` coordinate seven is used by every dense QKV row, and its bias is already fixed at zero, making absorption less constrained.

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
class FourPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with four absorbed biases and its final scale fixed at one."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        full_weight = self.weight.detach()
        self.bias = nn.Parameter(full_bias[:-4].clone())
        self.weight = nn.Parameter(full_weight[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 4))
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE