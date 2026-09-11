MECHANISM: Third pre-MLP LayerNorm-bias absorption gauge

HYPOTHESIS: Fixing a third `ln2.bias` coordinate will reduce the verified 1,608-parameter model to 1,607 parameters while retaining at least 99% accuracy, because the retained full `fc1.bias` can absorb every constant contribution from that LayerNorm offset.

INTENDED_EDIT: Retain five rather than six learned `ln2.bias` coordinates and reconstruct its final three coordinates as zeros.

EVIDENCE: Fixing the second `ln2.bias` coordinate achieved 99.94% accuracy at 1,608 parameters, directly qualifying another application of the same downstream-bias redundancy; alternative 1,608-parameter positional and MLP-row constraints failed substantially.

<<<<<<< SEARCH
class CompactLayerNormBias(nn.Module):
    """LayerNorm with two downstream-linear-absorbed bias coordinates fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-2].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(2)))
=======
class CompactLayerNormBias(nn.Module):
    """LayerNorm with three downstream-linear-absorbed bias coordinates fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-3].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(3)))
>>>>>>> REPLACE