MECHANISM: Fully fixed final LayerNorm bias

HYPOTHESIS: Fixing the last learned final LayerNorm bias coordinate at zero will reduce the model from 1,308 to 1,307 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Remove the final LayerNorm bias parameter entirely and use an eight-coordinate fixed-zero bias while preserving its learned scale.

EVIDENCE: Seven consecutive one-coordinate final LayerNorm bias reductions retained at least 99.99% accuracy, with the latest achieving 100% at 1,308 parameters; fixing the remaining coordinate is the smallest direct extension of that successful sequence.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with seven zero-fixed output-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 7)),
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with a fixed-zero output bias."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            None,
        )
>>>>>>> REPLACE