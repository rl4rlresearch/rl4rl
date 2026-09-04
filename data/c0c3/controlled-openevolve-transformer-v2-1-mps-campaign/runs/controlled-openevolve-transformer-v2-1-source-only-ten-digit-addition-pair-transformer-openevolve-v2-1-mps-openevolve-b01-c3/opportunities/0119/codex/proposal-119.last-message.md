MECHANISM: Triplet-and-two-pair secondary-head value-gain sharing

HYPOTHESIS: A 1,054-parameter transformer will maintain at least 99% accuracy because the verified 1,055-parameter `[a,b,b,c,c,d,d]` gain design achieved 99.88%, and tying its remaining singleton into the adjacent pair retains three independently learned feature gains.

INTENDED_EDIT: Adopt the verified cross-layer scalar-bias sharing and compress the secondary-head gain expansion to `[a,a,a,b,b,c,c]`, removing five deduplicated parameters from the current design.

EVIDENCE: The 1,055-parameter three-pair gain design reached 99.88%, while collapsing all seven gains to one scalar reached only 97.11%; the successful sequence of isolated adjacent ties motivates one further tie while preserving three feature groups.

<<<<<<< SEARCH
class ScalarBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with one scalar shared across bias coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.expand(self.linear.out_features)
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
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                self.secondary_value_gain,
            ),
            dim=0,
        )
=======
        y = att @ v
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :1].expand(-1, 3),
                self.secondary_value_gain[..., 1:2].expand(-1, 2),
                self.secondary_value_gain[..., 2:].expand(-1, 2),
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
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = ScalarBiasMeanZeroLinear(d_ff, d_model)
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