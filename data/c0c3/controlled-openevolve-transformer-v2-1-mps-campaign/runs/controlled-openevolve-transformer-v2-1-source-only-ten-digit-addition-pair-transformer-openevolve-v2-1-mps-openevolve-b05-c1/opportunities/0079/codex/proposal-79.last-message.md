MECHANISM: Common-mode-free attention output projection

HYPOTHESIS: Removing the eight row-common degrees of freedom from the attention output matrix will reduce the model from 1,272 to 1,264 parameters while preserving at least 99% accuracy, because the removed component adds only a position-dependent all-ones residual shift that every downstream LayerNorm eliminates.

INTENDED_EDIT: Replace the attention output projection’s dense 8×8 weight with a learned 7×8 matrix padded by a fixed-zero eighth output row, while retaining its unrestricted eight-coordinate bias and all existing bias sharing.

EVIDENCE: The 1,272-parameter design retained 100% accuracy after exact embedding gauge reductions, and the current MLP already uses this same seven-output-plus-zero-padding gauge successfully. Unlike the failed rotational query/key chart, this is a direct linear common-mode elimination already represented elsewhere in the verified architecture.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with five fixed scales and seven fixed bias coordinates."""
=======
class CommonModeFreeLinear(nn.Module):
    """Linear map with the unobservable common output row fixed to zero."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(self.linear(x), (0, 1)) + self.bias


class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with five fixed scales and seven fixed bias coordinates."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = CommonModeFreeLinear(d_model, d_model)
>>>>>>> REPLACE