MECHANISM: Zero-mean MLP-input quotient

HYPOTHESIS: Applying quotient coordinates to both attention and MLP inputs will produce a 1,281-parameter transformer with at least 99% accuracy, because each scale-fixed LayerNorm output is exactly zero-mean and the removed MLP weight direction is functionally inactive.

INTENDED_EDIT: Apply the qualified 1,292-parameter architecture and replace `fc1` with a seven-dimensional quotient-input linear map, removing eleven redundant weights.

EVIDENCE: The analogous QKV quotient removed 24 inactive weights and retained 99.59% accuracy; combining it with zero-mean `ln2` bias reached 99.96%, directly establishing the invariant needed for the same quotient at `fc1`.

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
        normal = torch.ones(embedding_dim) / math.sqrt(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("normal", normal, persistent=False)

        # A rank-seven matrix has only seven subspace degrees of freedom
        # after its invertible latent change of basis is absorbed by code.
        self.tilt = nn.Parameter(torch.zeros(rank))

    def projection_weight(self) -> torch.Tensor:
        return self.basis + torch.outer(self.normal, self.tilt)
=======
        self.register_buffer("basis", basis, persistent=False)

    def projection_weight(self) -> torch.Tensor:
        return self.basis
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
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = QuotientInputLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = QuotientBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE