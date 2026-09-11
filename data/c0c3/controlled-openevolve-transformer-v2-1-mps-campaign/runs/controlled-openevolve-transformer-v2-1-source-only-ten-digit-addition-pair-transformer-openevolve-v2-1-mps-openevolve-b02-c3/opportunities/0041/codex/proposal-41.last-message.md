MECHANISM: Single-coordinate Q/K scale gauge fixing

HYPOTHESIS: Fixing one nonzero key-projection coefficient while leaving its reciprocal query scale learned will reduce the qualified model from 1,038 to 1,037 parameters and retain at least 99% accuracy, because this removes one multiplicative Q/K factorization gauge without replacing the successfully optimized factorized attention.

INTENDED_EDIT: Add a quotient-input QKV projection whose first key coefficient is a fixed initialization-scale buffer, and use it in causal self-attention.

EVIDENCE: The 1,038-parameter factorized-attention design achieved 99.96%, while folding the attention factorizations into direct operators fell to 89.02%; this motivates an incremental gauge fix that preserves the factorized dot-product parameterization and its optimization behavior.

<<<<<<< SEARCH
class FactorizedTokenEmbedding(nn.Module):
    """Fixed-subspace low-rank token map shared with the output classifier."""
=======
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


class FactorizedTokenEmbedding(nn.Module):
    """Fixed-subspace low-rank token map shared with the output classifier."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = QuotientInputLinear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = GaugeFixedQKV(d_model)
>>>>>>> REPLACE