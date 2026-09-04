MECHANISM: Global residual-stream scale gauge fixing

HYPOTHESIS: Fixing the final LayerNorm’s eighth scale coordinate to one will reduce the model from 1,219 to 1,218 parameters while retaining at least 99% accuracy, because any positive value of that coordinate can be absorbed by globally scaling the tied embedding and residual-branch outputs while inversely scaling the final LayerNorm affine output.

INTENDED_EDIT: Store seven learned final-LayerNorm scale coordinates and reconstruct the eighth as the constant one; leave the protected decoding path and all other learned modules unchanged.

EVIDENCE: The current 1,219-parameter model achieved 99.97%, and the centered tied-embedding design previously preserved 100% accuracy after removing exact final-classifier gauges. This change removes another exact representational gauge rather than constraining attention, MLP, or embedding contrasts.

<<<<<<< SEARCH
class GaugeFixedFinalLayerNorm(nn.Module):
    """Final LayerNorm modulo the bias's centered-out common mode."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 1)),
            self.eps,
        )
=======
class GaugeFixedFinalLayerNorm(nn.Module):
    """Final LayerNorm modulo common bias and residual-scale gauges."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            F.pad(self.weight, (0, 1), value=1.0),
            F.pad(self.bias, (0, 1)),
            self.eps,
        )
>>>>>>> REPLACE