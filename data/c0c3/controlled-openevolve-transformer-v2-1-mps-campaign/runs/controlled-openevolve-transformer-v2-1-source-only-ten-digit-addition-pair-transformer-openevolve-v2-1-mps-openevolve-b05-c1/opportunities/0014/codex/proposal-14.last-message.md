MECHANISM: Quotient-space compression with extended convergence

HYPOTHESIS: The exact 36-parameter LayerNorm-nullspace compression will exceed 99% accuracy when trained for 8,000 steps, since it already reached 97.76% at 5,000 steps without narrowing the represented function class.

INTENDED_EDIT: Reparameterize QKV and `fc1` on the seven-dimensional mean-zero LayerNorm subspace, and increase the default training budget from 5,000 to 8,000 steps.

EVIDENCE: The 1,560-parameter joint nullspace design reached 97.76%, much closer to the threshold than other compressed failures, while the uncompressed model reached 99.99%; this suggests convergence, rather than representational capacity, is the most informative variable to test.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class MeanZeroInputLinear(nn.Module):
    """Linear map restricted to the mean-zero input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")

        basis = torch.zeros(in_features, in_features - 1)
        for j in range(in_features - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
        self.linear = nn.Linear(in_features - 1, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


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

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=5000)
=======
    p.add_argument("--train-steps", type=int, default=8000)
>>>>>>> REPLACE