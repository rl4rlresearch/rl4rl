MECHANISM: Complete pre-MLP LayerNorm bias absorption

HYPOTHESIS: Removing the final learned `ln2` bias coordinate will reduce the model from 1,602 to 1,601 parameters while retaining at least 99% accuracy, because the downstream learned `fc1` bias can absorb the LayerNorm bias contribution.

INTENDED_EDIT: Make `BiasAnchoredLayerNorm` entirely bias-free by removing its final bias parameter and passing no additive bias to layer normalization.

EVIDENCE: Seven consecutive `ln2` bias-coordinate removals retained at least 99% accuracy, culminating in 99.90% at 1,602 parameters; removing the sole remaining coordinate is the closest controlled extension.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with seven additive coordinates fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (x.size(-1),), self.weight, F.pad(self.bias, (0, 7)), 1e-5
        )
=======
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with its additive bias fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), self.weight, None, 1e-5)
>>>>>>> REPLACE