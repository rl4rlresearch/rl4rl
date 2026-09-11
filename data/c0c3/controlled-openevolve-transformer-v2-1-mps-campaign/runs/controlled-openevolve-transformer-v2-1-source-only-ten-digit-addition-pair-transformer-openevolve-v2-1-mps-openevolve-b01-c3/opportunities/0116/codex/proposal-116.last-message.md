MECHANISM: Pair-tied secondary-head value gain

HYPOTHESIS: The resulting 1,057-parameter transformer will maintain at least 99% accuracy because the verified 1,058-parameter model achieved 99.96%, and tying only the final pair of its seven secondary-head feature gains preserves six learned gain values instead of the single gain that failed at 97.11%.

INTENDED_EDIT: Adopt the verified 1,058-parameter mean-zero projections, triplet/quartet attention bias, and cross-layer scalar bias sharing, then remove one parameter by tying the final two coordinates of the secondary-head value adapter.

EVIDENCE: The 1,058-parameter reference reached 99.96%, while collapsing all seven secondary-head gains to one scalar reached only 97.11%; an isolated pair tie directly tests whether most feature-wise specialization can be retained with one fewer parameter.

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
class ScalarBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map reusing one scalar from a shared bias parameter."""

    def __init__(
        self, in_features: int, out_features: int, shared_bias: nn.Parameter
    ):
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = shared_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias[:1].expand(self.linear.out_features)
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 2))
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 1)
        )
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(2))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :-1],
                self.secondary_value_gain[..., -1:].expand(-1, 2),
            ),
            dim=-1,
        )
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                secondary_value_gain,
            ),
            dim=0,
        )
        y = (
            y * value_gain.view(1, self.n_head, 1, self.output_dim)
        ).sum(dim=1) / math.sqrt(self.n_head)
        bias = torch.cat(
            (self.output_bias[:1].expand(3), self.output_bias[1:].expand(4))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = PairTiedBiasMeanZeroLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        shared_bias: nn.Parameter,
    ):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = ScalarBiasMeanZeroLinear(d_ff, d_model, shared_bias)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(
            cfg.d_model, cfg.d_ff, cfg.dropout, self.attn.output_bias
        )
>>>>>>> REPLACE