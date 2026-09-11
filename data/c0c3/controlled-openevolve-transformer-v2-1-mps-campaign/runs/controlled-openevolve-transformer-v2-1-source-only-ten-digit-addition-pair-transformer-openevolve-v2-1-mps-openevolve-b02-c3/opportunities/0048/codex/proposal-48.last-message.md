MECHANISM: Final-classifier latent-bias coordinate fixing

HYPOTHESIS: Starting from the qualified four-key-fix architecture, fixing one zero-initialized final latent-bias coordinate at zero will produce a 1,033-parameter transformer with at least 99% accuracy because the trainable token codes, remaining four latent-bias coordinates, and upstream learned projections retain output-calibration freedom.

INTENDED_EDIT: Apply two fixed key-projection coefficients per attention head, then replace the five-parameter final latent bias with four learned coordinates and one fixed zero coordinate.

EVIDENCE: The 1,034-parameter four-key-fix design achieved 99.65%; three failed 1,033-parameter attempts modified attention routing or Q/K optimization, motivating an equally narrow reduction in the separate output-calibration path while preserving the qualified model’s initial function.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with one reciprocal Q/K scale fixed."""

    def __init__(self, d_model: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_index = d_model * in_features
        self.coeff = nn.Parameter(
            torch.empty(out_features * in_features - 1)
        )
        nn.init.normal_(self.coeff, mean=0.0, std=0.02)
        self.register_buffer(
            "fixed_coeff", torch.tensor([0.02]), persistent=False
        )

        basis = torch.zeros(d_model, in_features)
        for col in range(in_features):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_weight = torch.cat(
            (
                self.coeff[: self.fixed_index],
                self.fixed_coeff,
                self.coeff[self.fixed_index :],
            )
        )
        weight = flat_weight.view(self.out_features, self.in_features)
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
        return F.linear(quotient_x, weight)
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with two reciprocal Q/K scales fixed per head."""

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

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(
            (d_model + channel) * in_features
            for channel in sorted(selected_key_channels)
        )
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model)
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim)
        )
=======
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = x + self.final_bias @ self.token_emb.basis.transpose(0, 1)
=======
        final_bias = F.pad(self.final_bias, (0, 1))
        x = x + final_bias @ self.token_emb.basis.transpose(0, 1)
>>>>>>> REPLACE