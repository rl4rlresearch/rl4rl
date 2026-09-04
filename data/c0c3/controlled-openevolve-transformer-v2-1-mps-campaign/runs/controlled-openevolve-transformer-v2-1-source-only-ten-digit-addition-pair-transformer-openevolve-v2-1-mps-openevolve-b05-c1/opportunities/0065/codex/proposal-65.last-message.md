MECHANISM: Single fixed-coordinate final LayerNorm scale

HYPOTHESIS: Fixing one final LayerNorm scale coordinate at its initialized value of one while retaining the essential learned bias will reduce the model from 1,308 to 1,307 parameters and preserve at least 99% accuracy.

INTENDED_EDIT: Learn seven final LayerNorm scale coordinates, append a fixed-one eighth coordinate, and leave the successful one-coordinate learned bias unchanged.

EVIDENCE: The current 1,308-parameter model achieved 100% accuracy after seven incremental output-only bias restrictions, while removing the last adaptive bias collapsed accuracy to 29.11%; this tests an orthogonal, initialization-preserving output-only restriction without sacrificing that essential bias.

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
>>>>>>> REPLACE