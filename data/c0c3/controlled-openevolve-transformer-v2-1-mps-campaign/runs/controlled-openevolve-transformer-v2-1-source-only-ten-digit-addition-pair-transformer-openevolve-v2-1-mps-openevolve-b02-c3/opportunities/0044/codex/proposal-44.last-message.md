MECHANISM: Symmetric fourth reciprocal Q/K scale gauge fix

HYPOTHESIS: Fixing the second key-channel coefficient in the remaining attention head will produce a 1,034-parameter transformer with at least 99% accuracy because its corresponding learned query channel and bias preserve the reciprocal scaling freedom.

INTENDED_EDIT: Apply the qualified affine-free pre-MLP normalization and fixed positional amplitude, then fix two nonzero key-projection coefficients per attention head.

EVIDENCE: The 1,036-parameter per-head gauge-fixed design achieved 99.99%, and extending it to three fixed coefficients produced a qualified 1,035-parameter model at 99.78%; symmetrically fixing the second channel of the other head is the narrowest supported reduction.

<<<<<<< SEARCH
class FactorizedTokenEmbedding(nn.Module):
    """Fixed-subspace low-rank token map shared with the output classifier."""
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


class FactorizedTokenEmbedding(nn.Module):
    """Fixed-subspace low-rank token map shared with the output classifier."""
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