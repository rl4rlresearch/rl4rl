MECHANISM: Triple-pair-tied MLP output bias

HYPOTHESIS: The resulting 1,066-parameter transformer will maintain at least 99% accuracy because the verified 1,067-parameter model reached 99.82%, and two successive MLP output-bias pair ties already preserved 99%+ accuracy.

INTENDED_EDIT: Adopt the verified 1,067-parameter mean-zero attention and MLP factorization, then remove one additional parameter by tying a third pair of MLP output-bias coordinates.

EVIDENCE: The 1,067-parameter mean-zero MLP-input design achieved 99.82%; independently, the second MLP output-bias pair tie produced a 1,079-parameter model with 99.95%, motivating one more isolated tie while preserving all learned matrices, normalization, and attention routing.

<<<<<<< SEARCH
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
class MeanZeroInputLinear(nn.Module):
    """Bias-free linear map restricted to mean-zero input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features - 1, out_features, bias=False)
        self.register_buffer("basis", mean_zero_basis(in_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


class ThreePairTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with three adaptively shared bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 7:
            raise ValueError("out_features must be at least seven")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.free_bias[:-3], self.free_bias[-3:].repeat_interleave(2))
        )
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.query_dim = 1
        self.address_scale = nn.Parameter(torch.full((1,), 0.02))
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
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
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 2))
        self.register_buffer(
            "output_basis", mean_zero_basis(d_model), persistent=False
        )
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        address = self.address_scale * x[..., :1]
        v = self.value(x)

        q = address.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        k = address.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        q = q + self.q_bias.view(1, 1, 1, 1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        bsz, seqlen, _ = x.shape
        address = self.address_scale * x[..., :1]
        v = self.value(x)

        q = address.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        k = address.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        q = q + self.q_bias.view(1, 1, 1, 1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.unsqueeze(2).expand(-1, -1, self.n_head, -1).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y
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
        y = (y + bias) @ self.output_basis.transpose(0, 1)
        y = self.resid_drop(y)
        return y
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = PairwiseTiedBiasLinear(d_model, d_ff)
        self.fc2 = MeanZeroLinear(d_ff, d_model)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = ThreePairTiedBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE