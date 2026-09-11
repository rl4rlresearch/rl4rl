MECHANISM: Two-coordinate MLP preactivation bias gauge fixing

HYPOTHESIS: Fixing a second coordinate of the pre-MLP LayerNorm bias at zero will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because both fixed shifts are exactly absorbable by the learned `fc1` bias.

INTENDED_EDIT: Extend `MLPAnchoredLayerNorm` from one fixed-zero bias coordinate to two, preserving its zero initialization and all other architecture and training settings.

EVIDENCE: Fixing the first pre-MLP LayerNorm bias coordinate achieved 99.95% accuracy at 1619 parameters; applying one additional instance of the same exact redundancy is the smallest controlled next reduction.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with one shift coordinate absorbed by the following MLP bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias_rest.new_zeros(1), self.bias_rest))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
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
>>>>>>> REPLACE