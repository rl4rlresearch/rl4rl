MECHANISM: Pre-attention shift-to-bias gauge fixing

HYPOTHESIS: Fixing pre-attention LayerNorm bias coordinate 3 at zero will reduce the model from 1608 to 1607 parameters while retaining at least 99% accuracy, because its query and value shifts are absorbable by the learned attention biases, while its constant key shift cancels in the softmax.

INTENDED_EDIT: Replace the first block LayerNorm with an equivalent parameterization whose bias coordinate 3 is fixed at zero.

EVIDENCE: Coordinate 3 has been the strongest stable anchor: fixing its pre-MLP bias retained 99.93%, and its gain also remained fixed in subsequent passing designs. This tests a distinct exact attention-side redundancy without further constraining the sensitive pre-MLP coordinates 1, 2, or 5.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
=======
class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with shift coordinate 3 absorbed by attention biases."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class MLPAnchoredLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = AttentionAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE