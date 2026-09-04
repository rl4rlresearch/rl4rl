MECHANISM: Three-coordinate MLP preactivation bias gauge fixing

HYPOTHESIS: Fixing a third pre-MLP LayerNorm bias coordinate at zero will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because all three fixed shifts are exactly absorbable by the learned `fc1` bias.

INTENDED_EDIT: Extend `MLPAnchoredLayerNorm` from two fixed-zero bias coordinates to three while preserving zero initialization and all other architecture and training settings.

EVIDENCE: Fixing the first and then second pre-MLP LayerNorm bias coordinates achieved 99.95% at 1619 parameters and 99.92% at 1618 parameters; extending the same exact redundancy by one coordinate is the smallest controlled next reduction.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with two shift coordinates absorbed by the following MLP bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias_rest.new_zeros(2), self.bias_rest))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with three shift coordinates absorbed by the following MLP bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias_rest.new_zeros(3), self.bias_rest))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE