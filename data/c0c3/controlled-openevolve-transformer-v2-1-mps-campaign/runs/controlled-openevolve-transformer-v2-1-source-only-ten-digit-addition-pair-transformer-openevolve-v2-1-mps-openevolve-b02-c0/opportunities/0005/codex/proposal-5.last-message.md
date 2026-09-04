MECHANISM: Incremental pre-attention bias anchoring

HYPOTHESIS: Fixing a second coordinate of the first LayerNorm bias at zero will reduce the model from 1,643 to 1,642 parameters while retaining at least 99% accuracy, because the following learned QKV bias can absorb the omitted constant offsets.

INTENDED_EDIT: Store six learned bias coordinates in the pre-attention LayerNorm and append two fixed-zero coordinates.

EVIDENCE: Fixing one LayerNorm bias coordinate reduced the model to 1,643 parameters and achieved 99.95% accuracy, while larger structural reductions failed; this supports testing the smallest incremental extension of the successful redundancy removal.

<<<<<<< SEARCH
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with one fixed-zero bias coordinate."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, (x.size(-1),), self.weight, bias, 1e-5)
=======
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two fixed-zero bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), self.weight, bias, 1e-5)
>>>>>>> REPLACE