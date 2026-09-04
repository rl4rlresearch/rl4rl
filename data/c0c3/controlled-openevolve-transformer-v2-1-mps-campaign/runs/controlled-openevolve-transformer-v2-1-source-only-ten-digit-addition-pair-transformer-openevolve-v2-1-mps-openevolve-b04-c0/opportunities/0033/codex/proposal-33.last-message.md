MECHANISM: Third pre-MLP scale-to-weight gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm gain coordinate 1 to one will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because its scale is absorbable by column 1 of `fc1.weight`, its bias is already fixed at zero, and the initialization is unchanged.

INTENDED_EDIT: Remove gain coordinate 1 from the learned parameter vector and reconstruct gains 1, 3, and 7 as fixed ones during the forward pass.

EVIDENCE: Fixing gain coordinates 7 and 3 successively retained 99.97% and 99.93% accuracy; coordinate 1 is the strongest controlled next candidate because its LayerNorm bias has also remained successfully fixed throughout every passing bias-anchor design.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with four fixed shifts and gains 3 and 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 2))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat(
            (self.weight_rest[:3], ones(1), self.weight_rest[3:], ones(1))
        )
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with four fixed shifts and gains 1, 3, and 7 absorbed by the MLP."""

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
                self.weight_rest[:1],
                ones(1),
                self.weight_rest[1:2],
                ones(1),
                self.weight_rest[2:],
                ones(1),
            )
        )
>>>>>>> REPLACE