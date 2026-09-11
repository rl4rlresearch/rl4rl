MECHANISM: Full-channel cross-head key-feature sharing

HYPOTHESIS: The resulting 669-parameter transformer will retain at least 99% accuracy because Reference Design 1 achieved 99.88% after tying two corresponding coefficients of one key channel, and this patch ties only the third coefficient of that same channel while preserving head-specific queries, remaining keys, and learned routing.

INTENDED_EDIT: Reproduce the qualified 670-parameter design’s MLP bias fixing, relative-bias sharing, and two key ties, then deduplicate the adjacent third key coefficient.

EVIDENCE: Reference Design 2 achieved 99.96% with one cross-head key tie and Reference Design 1 achieved 99.88% with two; extending that successful sequence is better supported than removing query or final MLP biases, which collapsed accuracy to 45.62% and 19.51%.

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
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and three V shears fixed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
        selected_indices = {
            (d_model + channel) * in_features
            for channel in selected_key_channels
        }
        selected_indices.update(
            (2 * d_model + offset) * in_features
            for offset in range(min(3, head_dim))
        )
        shear_indices = {
            (2 * d_model) * in_features + offset
            for offset in (1, 2)
        }
        shear_indices.add(
            (2 * d_model + 1) * in_features + 2
        )
        selected_indices.update(shear_indices)

        self.in_features = in_features
=======
class GaugeFixedQKV(nn.Module):
    """Gauge-fixed QKV map sharing three learned key features across heads."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
        selected_indices = {
            (d_model + channel) * in_features
            for channel in selected_key_channels
        }
        selected_indices.update(
            (2 * d_model + offset) * in_features
            for offset in range(min(3, head_dim))
        )
        shear_indices = {
            (2 * d_model) * in_features + offset
            for offset in (1, 2)
        }
        shear_indices.add(
            (2 * d_model + 1) * in_features + 2
        )
        selected_indices.update(shear_indices)

        # Reuse the first three coefficients of the final key channel from
        # the first head in the corresponding channel of the second head.
        tied_source_indices = tuple(
            (d_model + head_dim - 1) * in_features + offset
            for offset in range(min(3, in_features))
        )
        tied_indices = tuple(
            (d_model + 2 * head_dim - 1) * in_features + offset
            for offset in range(min(3, in_features))
        )

        self.in_features = in_features
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.out_features = out_features
        self.fixed_indices = tuple(sorted(selected_indices))
        self.coeff = nn.Parameter(
            torch.empty(out_features * in_features - len(self.fixed_indices))
        )
=======
        self.out_features = out_features
        self.fixed_indices = tuple(sorted(selected_indices))
        self.tied_indices = tied_indices
        self.special_indices = tuple(
            sorted(self.fixed_indices + self.tied_indices)
        )
        self.tied_source_coeffs = tuple(
            source_index - sum(
                index < source_index for index in self.fixed_indices
            )
            for source_index in tied_source_indices
        )
        self.coeff = nn.Parameter(
            torch.empty(
                out_features * in_features
                - len(self.fixed_indices)
                - len(self.tied_indices)
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        learned_start = 0
        full_start = 0
        for fixed_index, fixed_coeff in zip(
            self.fixed_indices, self.fixed_coeff
        ):
            width = fixed_index - full_start
            pieces.append(
                self.coeff[learned_start : learned_start + width]
            )
            pieces.append(fixed_coeff.view(1))
            learned_start += width
            full_start = fixed_index + 1
        pieces.append(self.coeff[learned_start:])

        weight = torch.cat(pieces).view(
            self.out_features, self.in_features
        )
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        learned_start = 0
        full_start = 0
        fixed_offset = 0
        for special_index in self.special_indices:
            width = special_index - full_start
            pieces.append(
                self.coeff[learned_start : learned_start + width]
            )
            if special_index in self.tied_indices:
                tied_offset = self.tied_indices.index(special_index)
                source_coeff = self.tied_source_coeffs[tied_offset]
                pieces.append(
                    self.coeff[source_coeff : source_coeff + 1]
                )
            else:
                pieces.append(self.fixed_coeff[fixed_offset].view(1))
                fixed_offset += 1
            learned_start += width
            full_start = special_index + 1
        pieces.append(self.coeff[learned_start:])

        weight = torch.cat(pieces).view(
            self.out_features, self.in_features
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases and shares
        # one coefficient across its three longest remaining positions.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 8)
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