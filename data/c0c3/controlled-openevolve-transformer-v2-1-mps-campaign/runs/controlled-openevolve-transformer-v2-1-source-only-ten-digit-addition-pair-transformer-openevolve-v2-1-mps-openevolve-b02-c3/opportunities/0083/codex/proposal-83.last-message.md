MECHANISM: Third fixed sigmoid-gate intercept

HYPOTHESIS: A 809-parameter transformer will retain at least 99% accuracy because fixing a third gate bias at zero preserves all four gated nonlinear features and their input-dependent weights while leaving the qualified lexical and attention paths unchanged.

INTENDED_EDIT: Reconstruct the qualified 810-parameter design and fix the final three gated-MLP input-projection biases at zero instead of the final two.

EVIDENCE: The 810-parameter design achieved 99.82% after two gate intercepts were fixed, improving over the 812-parameter gated baseline’s 99.29%; the narrowest informative next reduction is one more intercept within this repeatedly successful mechanism.

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Distance zero is fixed because a per-head common bias cancels inside
        # softmax. The sparsely used maximum distance shares one learned bias.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
        self.farthest_relative_bias = nn.Parameter(torch.zeros(1))
=======
        # Distance zero cancels inside softmax, and the uniquely sparse
        # maximum distance is fixed at zero for both heads.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        learned_relative_bias = torch.cat(
            (
                self.relative_bias,
                self.farthest_relative_bias.expand(self.n_head, 1),
            ),
            dim=1,
        )
=======
        learned_relative_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=1,
        )
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