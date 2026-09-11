MECHANISM: Single fixed value-feature intercept

HYPOTHESIS: A 807-parameter transformer will retain at least 99% accuracy because fixing the fourth value-feature bias at zero preserves all four gated nonlinear features, every input-dependent value and gate weight, and the qualified lexical and attention paths.

INTENDED_EDIT: Learn only the first three gated-MLP input-projection biases, fixing the fourth value intercept and all four sigmoid-gate intercepts at zero.

EVIDENCE: The 808-parameter design achieved 99.91% after all four gate intercepts were fixed; its substantial margin motivates the narrowest adjacent reduction, removing one intercept while retaining every nonlinear feature and projection weight.

<<<<<<< SEARCH
class QuotientInputLinear(nn.Module):
    """Zero-mean-input linear map with its final four biases fixed."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")
        if bias and out_features < 5:
            raise ValueError("biased output must have at least five features")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 4))
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
        bias = None if self.bias is None else F.pad(self.bias, (0, 4))
        return F.linear(quotient_x, self.coeff.weight, bias)
=======
class QuotientInputLinear(nn.Module):
    """Zero-mean-input linear map with its final five biases fixed."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")
        if bias and out_features < 6:
            raise ValueError("biased output must have at least six features")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 5))
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
        bias = None if self.bias is None else F.pad(self.bias, (0, 5))
        return F.linear(quotient_x, self.coeff.weight, bias)
>>>>>>> REPLACE