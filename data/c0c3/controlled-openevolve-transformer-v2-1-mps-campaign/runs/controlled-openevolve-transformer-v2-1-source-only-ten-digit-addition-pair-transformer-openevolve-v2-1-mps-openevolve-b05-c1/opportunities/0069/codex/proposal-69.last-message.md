MECHANISM: Fourth coordinate-specific final LayerNorm scale restriction

HYPOTHESIS: Fixing final LayerNorm scale coordinate 3 at one, while retaining learned coordinate 5, will reduce the model from 1,305 to 1,304 parameters and preserve at least 99% accuracy.

INTENDED_EDIT: Learn final LayerNorm scale coordinates 0–2 and 5, fix coordinates 3–4 and 6–7 at one, and preserve the essential learned bias.

EVIDENCE: Fixing coordinate 4 in addition to coordinates 6–7 retained 100% accuracy at 1,305 parameters, while fixing coordinate 5 caused accuracy to fall to 75.89%; testing adjacent coordinate 3 is the smallest informative extension of the successful coordinate-specific restriction.

<<<<<<< SEARCH
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
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with four fixed scales and seven fixed bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 4))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:-1],
                self.weight.new_ones(2),
                self.weight[-1:],
                self.weight.new_ones(2),
            )
        )
>>>>>>> REPLACE