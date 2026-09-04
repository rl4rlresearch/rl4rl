MECHANISM: Zero-shift MLP LayerNorm gain absorption

HYPOTHESIS: Fixing MLP LayerNorm gain coordinate 1 at one will produce a 1575-parameter model with at least 99% accuracy because its shift is already fixed at zero, so the gain is exactly absorbable into `fc1.weight` column 1 without changing initialization.

INTENDED_EDIT: Remove the learned MLP LayerNorm gain for coordinate 1 and reconstruct it as one, leaving coordinate 2 as the sole learned gain.

EVIDENCE: Fixing attention LayerNorm gain coordinate 3 by the same zero-shift downstream-column absorption mechanism achieved 99.92% with 1576 parameters; the MLP LayerNorm already successfully fixes six other gains, making its zero-shift coordinate 1 the closest supported next reduction.

<<<<<<< SEARCH
    """LayerNorm with six fixed shifts and gains 3 through 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 6))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((ones(1), self.weight_rest, ones(5)))
=======
    """LayerNorm with six fixed shifts and only gain coordinate 2 learned."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 7))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((ones(2), self.weight_rest, ones(5)))
>>>>>>> REPLACE