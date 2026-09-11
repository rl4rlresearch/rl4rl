MECHANISM: Fourth pre-MLP scale-to-weight gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm gain coordinate 4 to one will reduce the model from 1613 to 1612 parameters while retaining at least 99% accuracy, because its scale is absorbable by `fc1.weight` and its distributed embedding anchor previously retained 99.94% accuracy.

INTENDED_EDIT: Remove gain coordinate 4 from the learned vector and reconstruct gains 3, 4, 5, and 7 as fixed ones during the forward pass.

EVIDENCE: Fixing gains 3, 5, and 7 achieved 99.95% accuracy at 1613 parameters; coordinate 4 is the strongest untested distributed-coordinate candidate because its embedding anchor previously achieved 99.94%, while coordinate 1 gain fixing collapsed to 27.21%.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with four fixed shifts and gains 3, 5, and 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 3))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat(
            (
                self.weight_rest[:3],
                ones(1),
                self.weight_rest[3:4],
                ones(1),
                self.weight_rest[4:],
                ones(1),
            )
        )
=======
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
>>>>>>> REPLACE