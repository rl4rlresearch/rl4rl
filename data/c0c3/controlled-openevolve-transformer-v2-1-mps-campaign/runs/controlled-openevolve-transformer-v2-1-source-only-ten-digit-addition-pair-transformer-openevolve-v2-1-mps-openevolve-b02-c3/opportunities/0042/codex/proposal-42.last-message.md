MECHANISM: Per-head Q/K scale gauge fixing

HYPOTHESIS: Applying the qualified fixed positional amplitude and fixing one key-projection coefficient in each attention head will produce a 1,036-parameter transformer with at least 99% accuracy, because each head has an independent reciprocal query/key scaling gauge and the first such fix already achieved 99.94%.

INTENDED_EDIT: Replace the 168 learned QKV coefficients with 166 learned coefficients plus one fixed nonpersistent key coefficient per head, and replace the learned positional-scale scalar with its qualified fixed value.

EVIDENCE: The 1,037-parameter design achieved 99.94% after fixing one key coefficient, while the fixed-position 1,038-parameter design achieved 99.96%; extending the same gauge fix to the untouched second head is the narrowest supported reduction.

<<<<<<< SEARCH
class FactorizedTokenEmbedding(nn.Module):
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with one reciprocal Q/K scale fixed per head."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(
            (d_model + head * head_dim) * in_features
            for head in range(n_head)
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
        self.pos_scale = nn.Parameter(torch.tensor(0.02))
=======
        self.register_buffer("pos_scale", torch.tensor(0.02), persistent=False)
>>>>>>> REPLACE