MECHANISM: Coordinate-specific final LayerNorm scale swap

HYPOTHESIS: Fixing final LayerNorm scale coordinate 1 while restoring learned coordinate 2 will reduce the model from 1,304 to 1,303 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Learn final LayerNorm scale coordinates 0, 2, and 5; fix coordinates 1, 3–4, and 6–7 at one; preserve the essential learned bias.

EVIDENCE: Fixing coordinate 2 caused accuracy to fall to 55.30%, while fixing coordinates 3–4 and 6–7 achieved 100% accuracy at 1,304 parameters. Restoring coordinate 2 and testing adjacent coordinate 1 isolates whether the failure was coordinate-specific.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE