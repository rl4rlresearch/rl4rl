MECHANISM: Bidirectionally tied gated-MLP directions

HYPOTHESIS: A 660-parameter transformer will retain at least 99% accuracy because each gated MLP unit can reuse its learned value/gate input directions to synthesize its output direction, eliminating 12 parameters while preserving the qualified 672-parameter model’s lexical, attention-routing, normalization, gating, and essential output-bias capacity.

INTENDED_EDIT: Reconstruct the qualified 672-parameter attention backbone, then replace the independent 20-scalar MLP output matrix with eight learned mixing coefficients that combine each hidden unit’s value and gate input directions.

EVIDENCE: The 672-parameter architecture achieved 99.52%, while deleting its last MLP output bias collapsed to 33.33% and reductions in lexical, normalization, and query capacity also failed. This motivates preserving those load-bearing components while challenging the untested assumption that the gated MLP requires input and output directions learned independently.

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = QuotientOutputLinear(d_model, d_model)
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head also fixes its third- and fourth-longest
        # distances. The second head fixes its next six longest biases,
        # shares its three longest remaining positions, and separately
        # shares the adjacent pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 11)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head retains all other distance coefficients.
        # The second head fixes its next six longest biases, shares its
        # three longest remaining positions, and separately shares the
        # adjacent pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 9)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 1))
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 2
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 2 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                self.relative_bias.new_zeros(2),
                second_head_bias[:-2],
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
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    """Compact multiplicatively gated nonlinear sublayer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff, bias=False)
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=4
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        hidden = F.gelu(value) * torch.sigmoid(gate)
        return self.drop(self.fc2(hidden))
=======
class MLP(nn.Module):
    """Gated sublayer tying each output direction to its input directions."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        if d_model < 6:
            raise ValueError("d_model must leave one learned output bias")

        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff, bias=False)

        initial_mix = torch.zeros(d_ff, 2)
        initial_mix[:, 0] = 1.0
        self.output_mix = nn.Parameter(initial_mix)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 5))

        basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("output_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        hidden = F.gelu(value) * torch.sigmoid(gate)

        value_weight, gate_weight = self.fc1.coeff.weight.chunk(2, dim=0)
        output_directions = (
            self.output_mix[:, :1] * value_weight
            + self.output_mix[:, 1:] * gate_weight
        )
        output_bias = F.pad(self.output_bias, (0, 4))
        output_coeff = F.linear(
            hidden, output_directions.transpose(0, 1), output_bias
        )
        output = output_coeff @ self.output_basis.transpose(0, 1)
        return self.drop(output)
>>>>>>> REPLACE