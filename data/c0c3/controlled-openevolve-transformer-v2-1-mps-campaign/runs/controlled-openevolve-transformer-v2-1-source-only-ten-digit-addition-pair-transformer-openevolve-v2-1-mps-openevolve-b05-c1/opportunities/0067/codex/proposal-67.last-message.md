MECHANISM: Third fixed-coordinate final LayerNorm scale

HYPOTHESIS: Fixing a third final LayerNorm scale coordinate at its initialized value of one will reduce the model from 1,306 to 1,305 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Learn five final LayerNorm scale coordinates instead of six, append three fixed-one coordinates, and preserve the essential one-coordinate learned bias.

EVIDENCE: Two consecutive one-coordinate final LayerNorm scale reductions retained at least 99.99% accuracy, most recently achieving 100% at 1,306 parameters; extending the same successful output-only restriction by one coordinate is the smallest informative next change.

<<<<<<< SEARCH
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
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with three fixed scales and seven fixed bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 3))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            F.pad(self.weight, (0, 3), value=1.0),
            F.pad(self.bias, (0, 7)),
        )
>>>>>>> REPLACE