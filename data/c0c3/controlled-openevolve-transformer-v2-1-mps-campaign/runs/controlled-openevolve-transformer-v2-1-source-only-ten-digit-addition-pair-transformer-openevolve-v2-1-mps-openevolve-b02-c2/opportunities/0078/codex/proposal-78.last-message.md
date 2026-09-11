MECHANISM: Fourth pre-MLP LayerNorm-bias absorption gauge

HYPOTHESIS: Fixing a fourth `ln2.bias` coordinate will reduce the model to 1,606 parameters while retaining at least 99% accuracy, because the retained full `fc1.bias` can absorb the removed LayerNorm offset’s contribution.

INTENDED_EDIT: Retain four learned `ln2.bias` coordinates and reconstruct the final four as zeros.

EVIDENCE: Fixing three `ln2.bias` coordinates achieved 99.42% accuracy at 1,607 parameters, after fixing two achieved 99.94% at 1,608; this directly supports testing one additional instance of the same downstream-bias redundancy.

<<<<<<< SEARCH
class CompactLayerNormBias(nn.Module):
    """LayerNorm with one downstream-linear-absorbed bias coordinate fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-1].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
=======
class CompactLayerNormBias(nn.Module):
    """LayerNorm with four downstream-linear-absorbed bias coordinates fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-4].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(4)))
>>>>>>> REPLACE