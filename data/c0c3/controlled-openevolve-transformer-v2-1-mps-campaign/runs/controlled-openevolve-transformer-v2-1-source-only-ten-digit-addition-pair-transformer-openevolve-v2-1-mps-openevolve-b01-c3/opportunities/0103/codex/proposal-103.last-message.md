MECHANISM: Mean-zero MLP input factorization

HYPOTHESIS: The 1,067-parameter transformer will maintain at least 99% accuracy because non-affine LayerNorm makes every MLP input feature-mean-zero, so replacing its learned 12×8 projection with an equivalent 12×7 projection removes only an unobservable input direction.

INTENDED_EDIT: Adopt the verified 1,079-parameter shared mean-zero attention value design with doubly pair-tied MLP output bias, then restrict the MLP input projection to the seven-dimensional mean-zero basis, removing twelve additional parameters.

EVIDENCE: The 1,079-parameter design achieved 99.95% accuracy, and the analogous mean-zero restriction of the attention value map reduced 1,087 parameters to 1,080 while retaining 99.79%; the MLP receives outputs from the same kind of non-affine LayerNorm.

<<<<<<< SEARCH
class MeanZeroLinear(nn.Module):
    """Linear map whose outputs lie in the feature-mean-zero subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1)
        self.register_buffer("basis", mean_zero_basis(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis.transpose(0, 1)


class PairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with one adaptively shared bias pair."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 3:
            raise ValueError("out_features must be at least three")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.free_bias[:-1], self.free_bias[-1:].expand(2)))
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
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
    """Bias-free linear map restricted to mean-zero input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features - 1, out_features, bias=False)
        self.register_buffer("basis", mean_zero_basis(in_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


class TwoPairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with two adaptively shared bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 5:
            raise ValueError("out_features must be at least five")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.free_bias[:-2], self.free_bias[-2:].repeat_interleave(2))
        )
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.query_dim = 1
        self.output_dim = d_model - 1
        self.address_scale = nn.Parameter(torch.full((1,), 0.02))
        self.value = nn.Linear(
            d_model, n_head * self.output_dim, bias=False
        )
=======
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if n_head < 2:
            raise ValueError("n_head must be at least two")

        self.n_head = n_head
        self.query_dim = 1
        self.output_dim = d_model - 1
        self.address_scale = nn.Parameter(torch.full((1,), 0.02))
        self.value = MeanZeroInputLinear(d_model, self.output_dim)
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        address = self.address_scale * x[..., :1]
        v = self.value(x)
=======
        bsz, seqlen, _ = x.shape
        address = self.address_scale * x[..., :1]
        v = self.value(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.output_dim).transpose(1, 2)
=======
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.unsqueeze(2).expand(-1, -1, self.n_head, -1).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.sum(dim=1)
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(2)))
=======
        y = att @ v
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                self.secondary_value_gain,
            ),
            dim=0,
        )
        y = (
            y * value_gain.view(1, self.n_head, 1, self.output_dim)
        ).sum(dim=1) / math.sqrt(self.n_head)
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(2)))
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = MeanZeroLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = TwoPairTiedBiasMeanZeroLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE