MECHANISM: Seventh fixed-coordinate final LayerNorm bias

HYPOTHESIS: Learning one final LayerNorm bias coordinate and fixing the remaining seven at zero will reduce the model from 1,309 to 1,308 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Reduce the final LayerNorm bias parameter from two coordinates to one and increase its fixed-zero padding from six coordinates to seven.

EVIDENCE: Six consecutive one-coordinate reductions of the final LayerNorm bias retained at least 99.99% accuracy, with the latest reaching 100% accuracy at 1,309 parameters; extending this successful output-only restriction by one coordinate is the smallest informative next change.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with six zero-fixed output-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 6)),
        )
=======
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
>>>>>>> REPLACE