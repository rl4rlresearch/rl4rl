MECHANISM: Distributed-coordinate pre-MLP scale gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm gain coordinate 5 to one will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because its scale is absorbable by `fc1.weight` and this tests a distinct redundancy from the failed coordinate-5 bias constraint.

INTENDED_EDIT: Remove gain coordinate 5 from the learned vector and reconstruct gains 3, 5, and 7 as fixed ones during the forward pass.

EVIDENCE: Gain coordinates 7 and 3 were fixed successfully at 99.97% and 99.93%, while coordinate 5’s distributed embedding anchor previously achieved 99.97%; its failed bias fixing result motivates testing whether sensitivity is specific to eliminating its shift rather than its scale.

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
>>>>>>> REPLACE