MECHANISM: Single-row LayerNorm gauge quotient

HYPOTHESIS: A 1,271-parameter model will retain at least 99% accuracy because one QKV output row can be parameterized on the seven-dimensional mean-zero LayerNorm subspace without reducing its learned function class, while leaving the verified 1,272-parameter model’s load-bearing positional biases and all other projections unchanged.

INTENDED_EDIT: Replace one of the 24 dense QKV rows with an orthonormally parameterized seven-coordinate row, removing exactly one learned parameter.

EVIDENCE: The 1,272-parameter design achieved 99.3%, whereas further positional sharing and MLP output-bias tying failed; the 1,237-parameter experiment supports testing LayerNorm-subspace redundancy more conservatively by quotienting only one projection row instead of all 36 simultaneously.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class SingleRowMeanZeroInputLinear(nn.Module):
    """Linear map with one row parameterized on the mean-zero input subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.first = nn.Linear(in_features - 1, 1, bias=False)
        self.rest = nn.Linear(in_features, out_features - 1, bias=False)
        self.register_buffer(
            "basis", mean_zero_basis(in_features), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.first(x @ self.basis)
        return torch.cat((first, self.rest(x)), dim=-1)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = SingleRowMeanZeroInputLinear(d_model, 3 * d_model)
>>>>>>> REPLACE