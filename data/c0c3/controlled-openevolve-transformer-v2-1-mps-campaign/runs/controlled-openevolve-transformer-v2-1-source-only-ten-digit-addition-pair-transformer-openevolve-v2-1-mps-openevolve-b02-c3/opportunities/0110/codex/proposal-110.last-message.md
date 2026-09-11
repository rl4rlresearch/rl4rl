MECHANISM: Gauge-free full-rank headwise value transport

HYPOTHESIS: A 673-parameter transformer will retain at least 99% accuracy because replacing the redundant rank-three V/output factorization with direct full-rank headwise quotient-space transport removes four parameters while increasing effective value-map rank, and the qualified 677-parameter three-way bias layout preserves routing.

INTENDED_EDIT: Use the qualified three-way second-head bias sharing, split learned Q/K routing from value transport, and replace each head’s factorized V-to-output path with one learned 5-by-5 map initialized at the original product scale.

EVIDENCE: The 677-parameter three-way-sharing design achieved 99.97%, while extending bias sharing to four positions collapsed to 50.28% and an extra QKV anchor collapsed to 16.69%; this motivates preserving the qualified routing computation and reducing parameters through a different, nonrestrictive value-transport parameterization.

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
        self.out_features = out_features
        self.fixed_indices = tuple(sorted(selected_indices))
        self.coeff = nn.Parameter(
            torch.empty(out_features * in_features - len(self.fixed_indices))
        )
        nn.init.normal_(self.coeff, mean=0.0, std=0.02)
        self.register_buffer(
            "fixed_coeff",
            torch.tensor(
                [
                    0.0 if index in shear_indices else 0.02
                    for index in self.fixed_indices
                ]
            ),
            persistent=False,
        )

        basis = torch.zeros(d_model, in_features)
        for col in range(in_features):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

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
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
        return F.linear(quotient_x, weight)
=======
class GaugeFixedQK(nn.Module):
    """Quotient-input Q/K map with per-head key-coordinate anchors."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 2 * d_model
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

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(sorted(selected_indices))
        self.coeff = nn.Parameter(
            torch.empty(out_features * in_features - len(self.fixed_indices))
        )
        nn.init.normal_(self.coeff, mean=0.0, std=0.02)
        self.register_buffer(
            "fixed_coeff",
            torch.full((len(self.fixed_indices),), 0.02),
            persistent=False,
        )

        basis = torch.zeros(d_model, in_features)
        for col in range(in_features):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

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
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
        return F.linear(quotient_x, weight)


class HeadwiseValueMixer(nn.Module):
    """Direct full-rank value transport in the zero-mean quotient space."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        quotient_dim = d_model - 1
        head_dim = d_model // n_head
        self.weight = nn.Parameter(
            torch.empty(n_head, quotient_dim, quotient_dim)
        )
        self.bias = nn.Parameter(torch.zeros(quotient_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.register_buffer(
            "transport_scale",
            torch.tensor(0.02 * math.sqrt(head_dim)),
            persistent=False,
        )

        basis = torch.zeros(d_model, quotient_dim)
        for col in range(quotient_dim):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(
        self, x: torch.Tensor, att: torch.Tensor
    ) -> torch.Tensor:
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
        context = torch.einsum("bhij,bjd->bhid", att, quotient_x)
        latent = self.transport_scale * torch.einsum(
            "bhtd,hod->bto", context, self.weight
        )
        latent = latent + self.bias
        return F.linear(latent, self.basis)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)

        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases and shares
        # its two longest remaining coefficients.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 7)
        )
=======
        self.qk = GaugeFixedQK(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.value_mixer = HeadwiseValueMixer(d_model, n_head)

        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases and shares
        # one coefficient across its three longest remaining positions.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 8)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
=======
        qk = self.qk(x)
        q, k = qk.chunk(2, dim=-1)
        q = q + self.q_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
                second_head_bias,
                second_head_bias[-1:],
                self.relative_bias.new_zeros(6),
=======
                second_head_bias,
                second_head_bias[-1:].expand(2),
                self.relative_bias.new_zeros(6),
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = self.value_mixer(x, att)
        y = self.resid_drop(y)
>>>>>>> REPLACE