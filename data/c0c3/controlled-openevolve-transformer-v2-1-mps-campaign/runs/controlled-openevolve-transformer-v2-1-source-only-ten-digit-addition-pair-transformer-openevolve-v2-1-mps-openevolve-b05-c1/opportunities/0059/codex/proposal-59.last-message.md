MECHANISM: Fifth fixed-coordinate final LayerNorm bias

HYPOTHESIS: Learning three final LayerNorm bias coordinates and fixing the remaining five at zero will reduce the model from 1,311 to 1,310 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Reduce the final LayerNorm bias parameter by one coordinate and increase its fixed-zero padding from four coordinates to five.

EVIDENCE: Four consecutive one-coordinate reductions of the final LayerNorm bias retained 100% accuracy, most recently reaching 1,311 parameters; extending this proven output-only restriction by one coordinate is the smallest informative next change.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with four zero-fixed output-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 4)),
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with five zero-fixed output-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 5)),
        )
>>>>>>> REPLACE