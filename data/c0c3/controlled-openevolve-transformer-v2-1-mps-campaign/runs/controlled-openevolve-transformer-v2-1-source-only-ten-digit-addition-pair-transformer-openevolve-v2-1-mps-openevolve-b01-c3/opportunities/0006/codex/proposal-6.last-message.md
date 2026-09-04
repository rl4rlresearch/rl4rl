MECHANISM: Orthonormal mean-zero input projection

HYPOTHESIS: Reparameterizing the QKV and MLP input projections over the seven-dimensional mean-zero LayerNorm subspace will reduce parameters from 1,596 to 1,560 while retaining at least 99% accuracy, because it preserves their full function classes and isotropic initialization on all inputs they receive.

INTENDED_EDIT: Add a parameter-efficient linear layer that projects affine-free LayerNorm outputs onto an orthonormal basis of their mean-zero subspace, then use it for QKV and MLP input projections.

EVIDENCE: The 1,596-parameter affine-free LayerNorm design achieved 99.81% accuracy, establishing that both projections receive mean-zero normalized inputs; unlike the failed `d_ff` reduction, this removes only unidentifiable input directions while preserving `d_ff=12`.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class MeanZeroLinear(nn.Module):
    """Linear map on the mean-zero subspace produced by affine-free LayerNorm."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        contrasts = torch.eye(in_features)[:, :-1] - torch.eye(in_features)[:, -1:]
        basis = torch.linalg.qr(contrasts, mode="reduced").Q
        self.register_buffer("basis", basis, persistent=False)
        self.linear = nn.Linear(in_features - 1, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = MeanZeroLinear(d_model, 3 * d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = MeanZeroLinear(d_model, d_ff)
>>>>>>> REPLACE