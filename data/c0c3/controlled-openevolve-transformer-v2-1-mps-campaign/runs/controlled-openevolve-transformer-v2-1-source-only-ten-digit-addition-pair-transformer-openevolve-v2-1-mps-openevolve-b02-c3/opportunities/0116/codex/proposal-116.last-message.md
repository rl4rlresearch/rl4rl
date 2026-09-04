MECHANISM: Two-coordinate MLP output-bias fixing

HYPOTHESIS: A 674-parameter transformer will retain at least 99% accuracy because it preserves the qualified 675-parameter attention-routing layout while fixing only one additional zero-initialized MLP output-bias coordinate.

INTENDED_EDIT: Reconstruct the qualified longest-bias triple and disjoint adjacent pair, then fix the final two quotient-space bias coefficients of the MLP output projection at zero.

EVIDENCE: The 675-parameter design achieved 99.97% after fixing one MLP output-bias coordinate while preserving the 676-parameter routing layout, directly motivating the next isolated bias-coordinate reduction without further constraining attention or value transport.

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
        # Distance zero and the two longest distances are fixed for both
        # heads; the second head's next six longest biases are also fixed.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 3) - 6)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases, shares
        # its three longest remaining positions, and separately shares the
        # adjacent pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 9)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_coeff = F.pad(self.relative_bias, (0, 6)).view(
            self.n_head, -1
        )
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                second_head_bias[:-2],
                second_head_bias[-2:-1].expand(2),
                second_head_bias[-1:].expand(3),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
=======
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=2
        )
>>>>>>> REPLACE