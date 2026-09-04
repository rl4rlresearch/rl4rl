MECHANISM: Second-head query-bias gauge fixing

HYPOTHESIS: A 670-parameter transformer will retain at least 99% accuracy because the qualified 671-parameter model already fixes one second-head query-bias coordinate, and the adjacent coordinate can likewise be absorbed by the learned second-head Q/K basis without changing attention routing, lexical codes, or MLP capacity.

INTENDED_EDIT: Reconstruct the qualified 671-parameter architecture, then fix the penultimate second-head query-bias coordinate at zero alongside the already fixed final coordinate.

EVIDENCE: The 671-parameter query-gauge design achieved 99.22%, while reductions to its remaining MLP bias, lexical codes, or final normalization failed; this motivates continuing the successful isolated reduction within the same query-bias gauge.

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
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)

        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases and shares
        # one coefficient across its three longest remaining positions.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 8)
        )
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 2))
        self.proj = QuotientOutputLinear(d_model, d_model)

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
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                second_head_bias,
                second_head_bias[-1:].expand(2),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
=======
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
            d_ff, d_model, fixed_bias_coeffs=4
        )
>>>>>>> REPLACE