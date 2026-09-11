MECHANISM: Second fixed-coordinate final LayerNorm bias

HYPOTHESIS: Fixing a second zero-initialized final LayerNorm bias coordinate will reduce the verified model from 1,314 to 1,313 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Learn six final LayerNorm bias coordinates instead of seven and pad the remaining two coordinates with zeros, while preserving all eight learned scale parameters and every training setting.

EVIDENCE: The immediately preceding 1,314-parameter design achieved 100% accuracy after fixing one final LayerNorm bias coordinate, while reductions that further coupled the sensitive MLP biases failed; extending the successful output-only restriction by exactly one coordinate is the smallest informative next change.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with one zero-fixed output-bias coordinate."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 1)),
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with two zero-fixed output-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 2)),
        )
>>>>>>> REPLACE