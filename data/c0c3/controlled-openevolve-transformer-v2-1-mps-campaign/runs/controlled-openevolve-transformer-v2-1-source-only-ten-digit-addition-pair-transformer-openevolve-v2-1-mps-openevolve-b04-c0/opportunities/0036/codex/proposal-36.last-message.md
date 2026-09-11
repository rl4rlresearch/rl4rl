MECHANISM: Fifth pre-MLP scale-to-weight gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm gain coordinate 6 to one will reduce the model from 1612 to 1611 parameters while retaining at least 99% accuracy, because its scale is absorbable by column 6 of `fc1.weight` and its distributed embedding anchor previously retained 99.72% accuracy.

INTENDED_EDIT: Remove gain coordinate 6 from the learned vector and reconstruct gains 3 through 7 as fixed ones during the forward pass.

EVIDENCE: Fixing gains 3, 4, 5, and 7 achieved 99.92% accuracy at 1612 parameters. Coordinate 6 is the only remaining untested distributed-anchor gain; its embedding anchor passed at 99.72%, while gain coordinate 1 previously collapsed to 27.21% and coordinate 2 has shown bias sensitivity.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with four fixed shifts and gains 3, 4, 5, and 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 4))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat(
            (
                self.weight_rest[:3],
                ones(3),
                self.weight_rest[3:],
                ones(1),
            )
        )
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with four fixed shifts and gains 3 through 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 5))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((self.weight_rest, ones(5)))
>>>>>>> REPLACE