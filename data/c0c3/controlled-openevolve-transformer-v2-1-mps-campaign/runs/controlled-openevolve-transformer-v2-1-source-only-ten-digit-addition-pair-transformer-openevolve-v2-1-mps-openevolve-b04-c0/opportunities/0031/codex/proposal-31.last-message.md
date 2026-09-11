MECHANISM: Pre-MLP scale-to-weight gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm gain coordinate 7 to one will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because its zero shift is already fixed and its gain is exactly absorbable by column 7 of `fc1.weight`.

INTENDED_EDIT: Replace gain coordinate 7 with a fixed one while preserving the existing fixed bias coordinates and initialization.

EVIDENCE: The current four-bias-anchor design achieved 99.93% at 1616 parameters, while adding bias coordinate 5 collapsed to 73.24%; fixing a gain already paired with zero bias tests a distinct exact redundancy without adding another shift constraint.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with shifts 0, 1, 3, and 7 absorbed by the MLP bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        bias = torch.cat(
            (zeros(2), self.bias_rest[:1], zeros(1), self.bias_rest[1:], zeros(1))
        )
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with four fixed shifts and gain 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 1))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        weight = torch.cat((self.weight_rest, self.weight_rest.new_ones(1)))
        bias = torch.cat(
            (zeros(2), self.bias_rest[:1], zeros(1), self.bias_rest[1:], zeros(1))
        )
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE