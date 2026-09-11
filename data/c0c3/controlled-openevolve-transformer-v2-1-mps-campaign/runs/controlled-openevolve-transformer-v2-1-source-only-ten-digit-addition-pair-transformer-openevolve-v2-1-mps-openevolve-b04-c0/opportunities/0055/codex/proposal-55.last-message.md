MECHANISM: Orthonormal final-LayerNorm common-shift quotient

HYPOTHESIS: Representing `fc2.bias` in an orthonormal seven-dimensional zero-sum basis will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy, because it removes the exact common-shift null direction without the asymmetric optimization geometry of fixing one output coordinate.

INTENDED_EDIT: Replace the ordinary eight-parameter `fc2` bias with seven learned orthonormal quotient coordinates whose reconstructed bias sums to zero.

EVIDENCE: Fixing `fc2.bias` coordinate 3 failed at 59% despite the common-shift invariance of the final LayerNorm; the current design reaches 99.91%, so a minimal-norm orthonormal parameterization directly tests whether that failure arose from the coordinate-anchored optimization geometry rather than loss of functional capacity.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = BiasAnchoredLinear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
class CommonShiftQuotientLinear(nn.Linear):
    """Linear layer quotienting the output-bias common-shift null direction."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_coords = nn.Parameter(torch.zeros(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = self.bias_coords.sum()
        root = math.sqrt(self.out_features)
        correction = (1.0 - 1.0 / root) * total / (self.out_features - 1)
        top = self.bias_coords - correction
        last = (-total / root).reshape(1)
        bias = torch.cat((top, last))
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = BiasAnchoredLinear(d_model, d_ff)
        self.fc2 = CommonShiftQuotientLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE