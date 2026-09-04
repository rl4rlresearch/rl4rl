MECHANISM: Downstream-affine LayerNorm bias absorption

HYPOTHESIS: Fixing one second-LayerNorm bias coordinate at zero will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy, because the following MLP input projection’s full bias can exactly absorb that coordinate.

INTENDED_EDIT: Replace the second LayerNorm with an equivalent gauge-fixed LayerNorm storing seven bias coordinates and padding the eighth with zero.

EVIDENCE: The 1622-parameter design reached 99.92% accuracy; unlike the failed MLP output-bias reduction, this removes an upstream bias coordinate that is directly and exactly representable by the full `fc1` bias.

<<<<<<< SEARCH
class MLP(nn.Module):
=======
class GaugeFixedLayerNormBias(nn.Module):
    """LayerNorm with one bias coordinate absorbed by a following affine layer."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 1)),
            self.eps,
        )


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = GaugeFixedLayerNormBias(cfg.d_model)
>>>>>>> REPLACE