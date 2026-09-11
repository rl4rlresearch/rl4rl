MECHANISM: Third fixed-coordinate final LayerNorm bias

HYPOTHESIS: Learning five final LayerNorm bias coordinates and fixing the remaining three at zero will reduce the model from 1,313 to 1,312 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Reduce the final LayerNorm bias parameter by one coordinate and increase fixed-zero padding from two coordinates to three.

EVIDENCE: The two immediately preceding output-only reductions each fixed one additional final LayerNorm bias coordinate and retained 100% accuracy, reaching 1,313 parameters; extending that successful sequence by exactly one coordinate is the smallest informative next change.

<<<<<<< SEARCH
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
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with three zero-fixed output-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 3)),
        )
>>>>>>> REPLACE