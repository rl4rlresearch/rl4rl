MECHANISM: Second fixed value-feature intercept

HYPOTHESIS: A 806-parameter transformer will retain at least 99% accuracy because fixing a second value-feature bias at zero preserves all four gated nonlinear features, every input-dependent value and gate weight, and the qualified lexical and attention paths.

INTENDED_EDIT: Reconstruct the qualified 807-parameter bias constraints and additionally fix the third value-feature intercept, leaving only the first two MLP input-projection biases learned.

EVIDENCE: The 807-parameter design achieved 99.93% after fixing the fourth value intercept and all gate intercepts; its margin motivates the narrowest adjacent reduction within the same repeatedly successful bias-fixing mechanism.

<<<<<<< SEARCH
class QuotientInputLinear(nn.Module):
    """Zero-mean-input linear map with its final three biases fixed."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")
        if bias and out_features < 4:
            raise ValueError("biased output must have at least four features")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 3))
            if bias
            else None
        )

        basis = torch.zeros(in_features, in_features - 1)
        for col in range(in_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
        bias = None if self.bias is None else F.pad(self.bias, (0, 3))
        return F.linear(quotient_x, self.coeff.weight, bias)
=======
class QuotientInputLinear(nn.Module):
    """Zero-mean-input linear map with its final six biases fixed."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")
        if bias and out_features < 7:
            raise ValueError("biased output must have at least seven features")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 6))
            if bias
            else None
        )

        basis = torch.zeros(in_features, in_features - 1)
        for col in range(in_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
        bias = None if self.bias is None else F.pad(self.bias, (0, 6))
        return F.linear(quotient_x, self.coeff.weight, bias)
>>>>>>> REPLACE