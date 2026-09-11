MECHANISM: Second fixed gate intercept

HYPOTHESIS: A 810-parameter transformer will retain at least 99% accuracy because fixing a second sigmoid-gate bias at zero preserves all four nonlinear features, every input-dependent gate, and the qualified lexical and attention paths.

INTENDED_EDIT: Reconstruct the qualified four-feature gated MLP, but learn only its first six input-projection biases and fix the final two gate intercepts at zero.

EVIDENCE: The 811-parameter design achieved 99.78% after fixing one gate intercept, while reductions to lexical rank or additional relative-attention biases failed; this motivates the narrowest adjacent one-parameter reduction within the successful gated-MLP mechanism.

<<<<<<< SEARCH
class QuotientInputLinear(nn.Module):
    """Linear map defined only on the zero-mean input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=bias)

        basis = torch.zeros(in_features, in_features - 1)
        for col in range(in_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(F.linear(x, self.basis.transpose(0, 1)))
=======
class QuotientInputLinear(nn.Module):
    """Zero-mean-input linear map with its final two output biases fixed."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")
        if bias and out_features < 3:
            raise ValueError("biased output must have at least three features")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 2))
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
        bias = None if self.bias is None else F.pad(self.bias, (0, 2))
        return F.linear(quotient_x, self.coeff.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = QuotientInputLinear(d_model, d_ff)
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class MLP(nn.Module):
    """Compact multiplicatively gated nonlinear sublayer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff)
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        hidden = F.gelu(value) * torch.sigmoid(gate)
        return self.drop(self.fc2(hidden))
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=7)
=======
    p.add_argument("--d-ff", type=int, default=4)
>>>>>>> REPLACE