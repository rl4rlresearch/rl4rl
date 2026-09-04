MECHANISM: Noncontiguous final LayerNorm scale restriction

HYPOTHESIS: Fixing scale coordinate 4 while restoring learned scale coordinate 5 will reduce the model from 1,306 to 1,305 parameters and retain at least 99% accuracy, because the failed trailing-coordinate reduction uniquely fixed coordinate 5, whose residual channel receives the attention-mean scalar reused by the MLP.

INTENDED_EDIT: Learn final LayerNorm scale coordinates 0–3 and 5, fix coordinate 4 and coordinates 6–7 at one, and preserve the essential learned bias.

EVIDENCE: Fixing trailing scale coordinates 6 and 7 retained 100% accuracy at 1,306 parameters, whereas additionally fixing coordinate 5 fell to 75.89%; testing a different third coordinate isolates whether that failure was coordinate-specific.

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
        weight = torch.cat(
            (
                self.weight[:-1],
                self.weight.new_ones(1),
                self.weight[-1:],
                self.weight.new_ones(2),
            )
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            F.pad(self.bias, (0, 7)),
        )
>>>>>>> REPLACE