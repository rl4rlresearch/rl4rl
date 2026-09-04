MECHANISM: Sixth coordinate-specific final LayerNorm scale restriction

HYPOTHESIS: Fixing final LayerNorm scale coordinate 0 at one while retaining learned coordinates 2 and 5 will reduce the model from 1,303 to 1,302 parameters and preserve at least 99% accuracy.

INTENDED_EDIT: Learn only final LayerNorm scale coordinates 2 and 5; fix coordinates 0–1, 3–4, and 6–7 at one while preserving the essential learned bias.

EVIDENCE: The 1,303-parameter design achieved 100% accuracy with learned coordinates 0, 2, and 5. Coordinate 2 previously proved essential, coordinate 5 is load-bearing, and coordinate 1 was safely fixed; testing coordinate 0 is the smallest remaining coordinate-specific reduction.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with five fixed scales and seven fixed bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 5))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:2],
                self.weight.new_ones(2),
                self.weight[2:],
                self.weight.new_ones(2),
            )
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with six fixed scales and seven fixed bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 6))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight.new_ones(2),
                self.weight[:1],
                self.weight.new_ones(2),
                self.weight[1:],
                self.weight.new_ones(2),
            )
        )
>>>>>>> REPLACE