MECHANISM: MLP preactivation bias gauge fixing

HYPOTHESIS: Fixing one coordinate of the second pre-LayerNorm bias at zero will reduce the model from 1620 to 1619 parameters while retaining at least 99% accuracy, because that coordinate’s effect is exactly absorbable by the existing `fc1` bias and the initialization remains unchanged.

INTENDED_EDIT: Add a LayerNorm variant with one fixed-zero bias coordinate and use it only before the MLP.

EVIDENCE: The current 1620-parameter design achieved 99.97% accuracy after successive exact gauge reductions; this tests another one-parameter functional redundancy while preserving the successful architecture and initialization.

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
=======
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


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = MLPAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE