MECHANISM: Third reciprocal Q/K scale gauge fix

HYPOTHESIS: Extending the qualified per-head Q/K gauge fixing to one additional key channel will produce a 1,035-parameter transformer with at least 99% accuracy, because its corresponding learned query channel and query bias retain the reciprocal scaling freedom.

INTENDED_EDIT: Apply the qualified affine-free pre-MLP normalization and fixed positional amplitude, then replace three learned QKV coefficients—one per head plus a second channel in the first head—with fixed nonzero initialization-scale coefficients.

EVIDENCE: The 1,036-parameter design achieved 99.99% accuracy after fixing one key coefficient per head; fixing one additional channel is the narrowest extension of that successful gauge reduction.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(F.linear(x, self.basis.transpose(0, 1)))


class FactorizedTokenEmbedding(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(F.linear(x, self.basis.transpose(0, 1)))


class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with three reciprocal Q/K scales fixed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        selected_key_channels = {
            head * head_dim for head in range(n_head)
        }
        selected_key_channels.add(1)

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


class FactorizedTokenEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = QuotientInputLinear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = QuotientBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_scale = nn.Parameter(torch.tensor(0.02))
=======
        self.register_buffer("pos_scale", torch.tensor(0.02), persistent=False)
>>>>>>> REPLACE