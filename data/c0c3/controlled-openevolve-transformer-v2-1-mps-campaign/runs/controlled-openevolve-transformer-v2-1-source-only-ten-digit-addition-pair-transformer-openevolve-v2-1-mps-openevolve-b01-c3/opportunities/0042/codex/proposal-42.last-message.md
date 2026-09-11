MECHANISM: LayerNorm-nullspace weight elimination

HYPOTHESIS: A 1,368-parameter model will retain at least 99% accuracy because the 36 removed qkv and MLP weight directions are annihilated by their preceding affine-free LayerNorms, while all effective projections and the verified three-quartet MLP thresholds remain learnable.

INTENDED_EDIT: Use the verified three-quartet MLP biases, then parameterize the qkv and first MLP projections only on the seven-dimensional mean-zero subspace produced by LayerNorm.

EVIDENCE: The 1,404-parameter three-quartet design achieved 99.57%. Shared values fell to 60.09% and shared query-key projections fell to 93.49%, showing effective attention maps must remain independent; this patch preserves those maps and removes only input-weight components that cannot affect mean-zero LayerNorm outputs.

<<<<<<< SEARCH
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the feature-mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


class AnchoredScaleLayerNorm(nn.Module):
=======
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the feature-mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


class MeanZeroInputLinear(nn.Module):
    """Linear map defined only on the feature-mean-zero input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features - 1, out_features, bias=bias)
        self.register_buffer("basis", mean_zero_basis(in_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


class AnchoredScaleLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with two learned-bias quartets and learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 8 or out_features % 2 != 0:
            raise ValueError("out_features must be even and at least eight")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 2 - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_biases = self.free_bias[:2].repeat_interleave(4)
        paired_biases = self.free_bias[2:].repeat_interleave(2)
        bias = torch.cat((quartet_biases, paired_biases))
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(MeanZeroInputLinear):
    """Mean-zero-input linear layer whose outputs form learned-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__(in_features, out_features, bias=False)
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.repeat_interleave(4)
        return F.linear(x @ self.basis, self.linear.weight, bias)
>>>>>>> REPLACE