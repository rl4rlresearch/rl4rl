MECHANISM: Fourth fixed-coordinate final LayerNorm bias

HYPOTHESIS: Learning four final LayerNorm bias coordinates and fixing the remaining four at zero will reduce the model from 1,312 to 1,311 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Reduce the final LayerNorm bias parameter by one coordinate and increase its fixed-zero padding from three coordinates to four.

EVIDENCE: The three consecutive final LayerNorm bias reductions each removed one parameter while retaining 100% accuracy, most recently reaching 1,312 parameters; extending the same successful output-only restriction by one coordinate is the smallest informative next change.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE