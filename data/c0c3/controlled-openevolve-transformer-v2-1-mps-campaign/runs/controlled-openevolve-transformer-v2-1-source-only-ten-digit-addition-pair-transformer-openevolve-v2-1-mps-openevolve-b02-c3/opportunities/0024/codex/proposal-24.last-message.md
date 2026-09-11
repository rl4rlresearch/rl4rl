MECHANISM: Zero-mean attention-input quotient

HYPOTHESIS: Replacing the attention QKV map’s eight-dimensional input with its seven-dimensional zero-mean quotient will reduce the verified model from 1,317 to 1,293 parameters while retaining at least 99% accuracy, because `GaugeFixedLayerNorm` always produces zero-mean vectors and the removed QKV weight direction therefore receives neither activation nor gradient.

INTENDED_EDIT: Add a quotient-input linear layer and use it for the learned QKV projection, removing 24 functionally inactive weights without changing attention behavior or decoding.

EVIDENCE: The current 1,317-parameter design achieved 100% accuracy after fixing both pre-attention and pre-MLP LayerNorm scales. Its pre-attention bias is also expressed entirely in a zero-mean basis, making the QKV input’s all-ones direction exactly unobservable; the orthonormal reparameterization preserves the initialized function distribution.

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
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = QuotientInputLinear(d_model, 3 * d_model, bias=False)
>>>>>>> REPLACE