MECHANISM: Single-coordinate second-pre-norm scale absorption

HYPOTHESIS: Fixing one `ln2` scale to one will reduce the model from 1,540 to 1,539 parameters while retaining at least 99% accuracy, because the following `fc1` weight column can absorb that coordinate’s learned scaling without reducing functional capacity.

INTENDED_EDIT: Replace `ln2` with a LayerNorm that learns seven scales and fixes the eighth to one, while leaving all other qualified components and training behavior unchanged.

EVIDENCE: Removing all eight `ln2` scales collapsed accuracy to 73.68%, whereas the current 1,540-parameter model achieved 99.99%; removing only one scale is the smallest controlled test of whether the failure resulted from simultaneous loss of all eight optimization degrees of freedom.

<<<<<<< SEARCH
class MLP(nn.Module):
=======
class OneFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with one scale absorbed by the following linear."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln2 = OneFixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE