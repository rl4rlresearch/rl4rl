MECHANISM: Eight-unit nonlinear carry bottleneck

HYPOTHESIS: A 1,004-parameter transformer will maintain at least 99% accuracy because successive MLP-width reductions to eleven, ten, and nine units achieved 100%, 99.89%, and 99.96%, respectively; reducing the verified nine-unit design by one more unit tests the same isolated 14-parameter reduction.

INTENDED_EDIT: Adopt the verified 1,018-parameter architecture and reduce its MLP hidden width from nine to eight units, preserving its full secondary-head gains, three-group attention bias, independent scalar MLP bias, routing, embeddings, and residual width.

EVIDENCE: The otherwise corresponding eleven-, ten-, and nine-unit models all exceeded 99% accuracy, with the nine-unit 1,018-parameter model reaching 99.96%; this repeated success makes another identical width reduction the clearest informative test.

<<<<<<< SEARCH
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
=======
class ScalarBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with one scalar shared across bias coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.expand(self.linear.out_features)
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 3)
        )
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(2))
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :1],
                self.secondary_value_gain[..., 1:2].expand(-1, 2),
                self.secondary_value_gain[..., 2:3].expand(-1, 2),
                self.secondary_value_gain[..., 3:].expand(-1, 2),
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
=======
        y = att @ v
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                self.secondary_value_gain,
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (self.output_bias[:1].expand(3), self.output_bias[1:].expand(4))
        )
=======
        bias = torch.cat(
            (
                self.output_bias[:1],
                self.output_bias[1:2].expand(2),
                self.output_bias[-1:].expand(4),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = ScalarBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.mlp = MLP(
            cfg.d_model, cfg.d_ff, cfg.dropout, self.attn.output_bias
        )
=======
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE