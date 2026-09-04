MECHANISM: Mean-zero input factorization for the attention value map

HYPOTHESIS: A 1,080-parameter transformer will maintain at least 99% accuracy because the value map receives non-affine LayerNorm outputs, whose all-ones input component is always zero, so removing that unidentifiable input direction preserves its effective learned function class.

INTENDED_EDIT: Replace the shared 8→7 value projection with a learned 7→7 projection over an orthonormal mean-zero input basis, removing seven parameters while preserving the head adapter and all routing mechanisms.

EVIDENCE: The current 1,087-parameter design achieved 99.98% accuracy; unlike prior failed reductions to positional biases or normalization, this removes only value weights multiplying a direction eliminated by the existing pre-attention LayerNorm.

<<<<<<< SEARCH
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


class PairTiedBiasMeanZeroLinear(MeanZeroLinear):
=======
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


class MeanZeroInputLinear(nn.Module):
    """Bias-free linear map restricted to mean-zero input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features - 1, out_features, bias=False)
        self.register_buffer("basis", mean_zero_basis(in_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


class PairTiedBiasMeanZeroLinear(MeanZeroLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.value = nn.Linear(d_model, self.output_dim, bias=False)
=======
        self.value = MeanZeroInputLinear(d_model, self.output_dim)
>>>>>>> REPLACE