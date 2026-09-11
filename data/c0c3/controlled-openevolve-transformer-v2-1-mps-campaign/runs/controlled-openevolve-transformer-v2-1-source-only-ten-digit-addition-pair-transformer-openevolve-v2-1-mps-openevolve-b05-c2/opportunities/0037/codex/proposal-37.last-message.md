MECHANISM: Query-key orthogonal gauge fixing

HYPOTHESIS: Fixing one query-weight coefficient through a shared orthogonal rotation of two query/key channels will produce a 1537-parameter model with at least 99% accuracy because it preserves initialized attention scores and removes only an exact attention-basis symmetry.

INTENDED_EDIT: Retain the qualified single mean-zero QKV row, rotate the first two query channels and their paired key channels so the first query coefficient is zero, and omit that fixed coefficient from the trainable parameterization.

EVIDENCE: The 1538-parameter single-row design achieved 99.77%, while extending its LayerNorm-input constraint to a second row fell to 96.45%; this tests a distinct exact query-key rotational gauge instead of repeating the disruptive row constraint.

<<<<<<< SEARCH
class SingleRowMeanZeroInputLinear(nn.Module):
    """Bias-free linear map with one LayerNorm-null row direction removed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        basis = _mean_zero_basis(in_features)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        discarded_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(discarded_bias, -bound, bound)

        self.weight_rows = nn.Parameter(full_weight[:-1])
        self.last_weight = nn.Parameter(full_weight[-1] @ basis)

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        with torch.no_grad():
            self.weight_rows.copy_(full_weight[:-1])
            self.last_weight.copy_(full_weight[-1] @ self.basis)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.basis @ self.last_weight
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)
=======
class SingleRowMeanZeroInputLinear(nn.Module):
    """QKV map with one input-null and one query-key rotation gauge fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        self.in_features = in_features
        self.out_features = out_features
        basis = _mean_zero_basis(in_features)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        discarded_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(discarded_bias, -bound, bound)

        fixed_weight = self._gauge_fix(full_weight)
        self.first_weight = nn.Parameter(fixed_weight[0, 1:])
        self.weight_rows = nn.Parameter(fixed_weight[1:-1])
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)

    def _gauge_fix(self, full_weight: torch.Tensor) -> torch.Tensor:
        fixed_weight = full_weight.clone()
        pivot = fixed_weight[:2, 0]
        radius = pivot.norm().clamp_min(torch.finfo(full_weight.dtype).tiny)
        cosine = pivot[1] / radius
        sine = -pivot[0] / radius
        rotation = torch.stack(
            (
                torch.stack((cosine, sine)),
                torch.stack((-sine, cosine)),
            )
        )
        fixed_weight[:2] = rotation @ fixed_weight[:2]
        key_start = self.in_features
        fixed_weight[key_start:key_start + 2] = (
            rotation @ fixed_weight[key_start:key_start + 2]
        )
        fixed_weight[0, 0] = 0.0
        return fixed_weight

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        fixed_weight = self._gauge_fix(full_weight)
        with torch.no_grad():
            self.first_weight.copy_(fixed_weight[0, 1:])
            self.weight_rows.copy_(fixed_weight[1:-1])
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)

    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (1, 0))
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                self.weight_rows,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)
>>>>>>> REPLACE