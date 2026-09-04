MECHANISM: Fifth-coordinate MLP output-bias fixing

HYPOTHESIS: A 671-parameter transformer will retain at least 99% accuracy because it preserves the qualified 672-parameter attention and weight architecture while fixing only the final remaining zero-initialized MLP output-bias coordinate.

INTENDED_EDIT: Add quotient-output bias-coordinate fixing, reconstruct the qualified four-coordinate fixing, and remove the sole remaining learned MLP output bias.

EVIDENCE: Successive MLP output-bias reductions achieved 99.97% at 675 parameters, 99.89% at 674, 99.71% at 673, and 99.52% at 672; this directly motivates testing the final bias coordinate without altering attention routing or learned weight matrices.

<<<<<<< SEARCH
class QuotientOutputLinear(nn.Module):
    """Linear map into the zero-mean feature subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")

        self.coeff = nn.Linear(in_features, out_features - 1, bias=bias)

        basis = torch.zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(x) @ self.basis.transpose(0, 1)
=======
class QuotientOutputLinear(nn.Module):
    """Linear map into the zero-mean feature subspace."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fixed_bias_coeffs: int = 0,
    ):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")
        if fixed_bias_coeffs < 0 or fixed_bias_coeffs >= out_features - 1:
            raise ValueError("invalid number of fixed bias coefficients")

        self.fixed_bias_coeffs = fixed_bias_coeffs
        self.coeff = nn.Linear(in_features, out_features - 1, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 1 - fixed_bias_coeffs))
            if bias
            else None
        )

        basis = torch.zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = (
            None
            if self.bias is None
            else F.pad(self.bias, (0, self.fixed_bias_coeffs))
        )
        return F.linear(x, self.coeff.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff, bias=False)
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff, bias=False)
        self.fc2 = QuotientOutputLinear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE