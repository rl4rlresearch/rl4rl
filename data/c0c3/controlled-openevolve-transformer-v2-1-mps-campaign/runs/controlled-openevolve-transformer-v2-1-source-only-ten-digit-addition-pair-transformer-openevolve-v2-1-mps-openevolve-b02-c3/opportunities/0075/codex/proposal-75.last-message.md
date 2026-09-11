MECHANISM: Residual first-head value/output shear gauge fixing

HYPOTHESIS: An 822-parameter transformer will retain at least 99% accuracy because fixing one first-head output-projection coefficient uses a residual value/output basis gauge without adding another fragile constraint to the qualified value pivot.

INTENDED_EDIT: Reconstruct the qualified 823-parameter QKV constraints, then fix the first coefficient of the attention output projection at zero, removing one learned parameter.

EVIDENCE: The 823-parameter lower-triangular value design achieved 99.83%, while extending its value-matrix constraints reached only 97.77%; moving the next gauge constraint to the coupled output side tests the remaining symmetry without further restricting that value pivot.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(x) @ self.basis.transpose(0, 1)


class QuotientInputLinear(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(x) @ self.basis.transpose(0, 1)


class GaugeFixedAttentionOutput(nn.Module):
    """Zero-mean attention output map with one shear-gauge entry fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")

        self.in_features = in_features
        self.out_features = out_features
        self.weight_coeff = nn.Parameter(
            torch.empty((out_features - 1) * in_features - 1)
        )
        self.bias_coeff = nn.Parameter(torch.zeros(out_features - 1))
        nn.init.normal_(self.weight_coeff, mean=0.0, std=0.02)
        self.register_buffer(
            "fixed_coeff", torch.tensor(0.0), persistent=False
        )

        basis = torch.zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.fixed_coeff.view(1), self.weight_coeff)
        ).view(self.out_features - 1, self.in_features)
        return F.linear(x, weight, self.bias_coeff) @ self.basis.transpose(0, 1)


class QuotientInputLinear(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K and two first-head V scales fixed."""

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
        selected_rows = {
            d_model + channel for channel in selected_key_channels
        }
        selected_rows.update(
            2 * d_model + offset for offset in range(min(2, head_dim))
        )

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(
            row * in_features for row in sorted(selected_rows)
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = QuotientOutputLinear(d_model, d_model)
=======
        self.proj = GaugeFixedAttentionOutput(d_model, d_model)
>>>>>>> REPLACE