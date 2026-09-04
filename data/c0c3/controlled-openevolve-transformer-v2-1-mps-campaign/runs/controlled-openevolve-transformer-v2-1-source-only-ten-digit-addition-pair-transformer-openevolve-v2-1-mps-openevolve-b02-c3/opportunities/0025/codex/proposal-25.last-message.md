MECHANISM: Zero-mean attention quotient with one-direction pre-MLP bias gauge fixing

HYPOTHESIS: Combining the verified quotient-input attention with a seven-direction zero-mean `ln2` bias will yield a 1,292-parameter transformer with at least 99% accuracy, because the removed all-ones bias component is exactly absorbable by `fc1.bias`.

INTENDED_EDIT: Use the verified seven-dimensional QKV input quotient, fix the pre-MLP LayerNorm scale, and retain seven learned zero-mean pre-MLP bias directions.

EVIDENCE: The 1,293-parameter quotient-attention design achieved 99.59% accuracy. Removing all `ln2` bias previously harmed optimization, so this preserves seven of eight bias directions while removing only one functionally redundant component.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(x) @ self.basis.transpose(0, 1)


class FactorizedTokenEmbedding(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(x) @ self.basis.transpose(0, 1)


class QuotientInputLinear(nn.Module):
    """Linear map defined only on the zero-mean input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=bias)

        basis = torch.zeros(in_features, in_features - 1)
        for col in range(in_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(F.linear(x, self.basis.transpose(0, 1)))


class FactorizedTokenEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias_coeff @ self.basis.transpose(0, 1)


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias_coeff @ self.basis.transpose(0, 1)


class QuotientBiasLayerNorm(nn.Module):
    """Scale-fixed LayerNorm with a zero-mean learned bias."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 2:
            raise ValueError("normalized_shape must be at least two")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 1))

        basis = torch.zeros(normalized_shape, normalized_shape - 1)
        for col in range(normalized_shape - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.bias_coeff @ self.basis.transpose(0, 1)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = QuotientInputLinear(d_model, 3 * d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = QuotientBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE