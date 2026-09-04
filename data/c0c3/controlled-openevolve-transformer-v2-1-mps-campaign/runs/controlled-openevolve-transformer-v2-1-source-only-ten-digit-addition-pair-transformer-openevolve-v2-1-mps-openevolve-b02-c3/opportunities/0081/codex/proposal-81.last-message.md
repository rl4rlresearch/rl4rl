MECHANISM: Single fixed gate intercept

HYPOTHESIS: An 811-parameter transformer will retain at least 99% accuracy because fixing one sigmoid-gate bias at its zero initialization preserves all four learned gated features, their input-dependent gates, and the qualified model’s attention and lexical capacity.

INTENDED_EDIT: Replace the gated MLP input projection’s final learned bias—the fourth gate intercept—with a fixed zero, removing one parameter while leaving every projection weight and the other seven biases learned.

EVIDENCE: The 812-parameter four-feature gated MLP achieved 99.29% accuracy; this motivates testing a one-parameter reduction inside that successful mechanism instead of reducing feature count, lexical rank, or additional relative-attention biases, which prior evidence identifies as fragile.

<<<<<<< SEARCH
class QuotientInputLinear(nn.Module):
    """Linear map defined only on the zero-mean input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=bias)
=======
class QuotientInputLinear(nn.Module):
    """Zero-mean-input linear map with its final output bias fixed at zero."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 1))
            if bias
            else None
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(F.linear(x, self.basis.transpose(0, 1)))
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
        bias = None if self.bias is None else F.pad(self.bias, (0, 1))
        return F.linear(quotient_x, self.coeff.weight, bias)
>>>>>>> REPLACE