MECHANISM: First-head Q/K coordinate-scale gauge fixing

HYPOTHESIS: A 663-parameter transformer will retain at least 99% accuracy because fixing the remaining first-head key-channel scale removes a Q/K rescaling redundancy while preserving the qualified 664-parameter model’s routing biases and representable attention scores.

INTENDED_EDIT: Reconstruct the qualified 664-parameter first-head quintet design, then fix one additional first-head key coefficient at 0.02; its learned query coordinate can absorb the reciprocal scale.

EVIDENCE: The 664-parameter quintet design achieved 99.57%, while both attempted 663-parameter relative-bias extensions failed; this tests an orthogonal attention gauge without further compressing the fragile routing biases.

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
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
        selected_indices = {
=======
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
        if head_dim > 2:
            selected_key_channels.add(2)
        selected_indices = {
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
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = QuotientOutputLinear(d_model, d_model)

        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent quintet immediately preceding them. The second
        # head fixes its next six longest biases, shares its three longest
        # remaining positions, and separately shares the adjacent pair
        # immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 16)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                second_head_bias,
                second_head_bias[-1:].expand(2),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 7
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 7 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias[:-1],
                first_head_bias[-1:].expand(5),
                self.relative_bias.new_zeros(3),
                second_head_bias[:-2],
                second_head_bias[-2:-1].expand(2),
                second_head_bias[-1:].expand(3),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff, bias=False)
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff, bias=False)
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=4
        )
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE