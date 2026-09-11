MECHANISM: LayerNorm nullspace parameterization

HYPOTHESIS: Reparameterizing the QKV and MLP-input linear maps on the seven-dimensional mean-zero subspace produced by non-affine LayerNorm will reduce parameters from 1,596 to 1,560 while retaining at least 99% accuracy.

INTENDED_EDIT: Add a linear layer whose fixed orthonormal basis removes the unobservable all-ones input-weight direction, then use it for QKV and `fc1`.

EVIDENCE: The 1,596-parameter design reached 99.99% accuracy after making both preceding LayerNorms non-affine; their outputs have zero channel mean, so 24 QKV and 12 `fc1` weight directions are functionally inactive and can be removed without narrowing either learned output space.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class MeanZeroInputLinear(nn.Module):
    """Linear map restricted to the mean-zero input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        seed = torch.eye(in_features)[:, : in_features - 1]
        seed[-1, :] = -1.0
        basis = torch.linalg.qr(seed, mode="reduced").Q
        self.register_buffer("basis", basis)
        self.weight = nn.Parameter(torch.empty(out_features, in_features - 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x @ self.basis, self.weight, self.bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
>>>>>>> REPLACE