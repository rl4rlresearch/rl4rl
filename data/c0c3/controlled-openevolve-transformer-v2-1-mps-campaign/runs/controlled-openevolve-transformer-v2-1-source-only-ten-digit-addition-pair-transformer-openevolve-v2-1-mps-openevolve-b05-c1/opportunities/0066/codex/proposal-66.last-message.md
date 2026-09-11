MECHANISM: Second fixed-coordinate final LayerNorm scale

HYPOTHESIS: Fixing a second final LayerNorm scale coordinate at its initialized value of one will reduce the model from 1,307 to 1,306 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Learn six final LayerNorm scale coordinates instead of seven, append two fixed-one coordinates, and preserve the essential one-coordinate learned bias.

EVIDENCE: Fixing the first final LayerNorm scale coordinate retained 99.99% accuracy at 1,307 parameters, while removing the remaining learned final bias collapsed accuracy; extending the successful scale restriction by exactly one coordinate is the smallest informative next change.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with one fixed scale and seven fixed bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            F.pad(self.weight, (0, 1), value=1.0),
            F.pad(self.bias, (0, 7)),
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with two fixed scales and seven fixed bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 2))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            F.pad(self.weight, (0, 2), value=1.0),
            F.pad(self.bias, (0, 7)),
        )
>>>>>>> REPLACE