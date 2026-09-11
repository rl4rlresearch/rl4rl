MECHANISM: Cross-layer scalar bias sharing

HYPOTHESIS: The 1,058-parameter transformer will maintain at least 99% accuracy because the verified 1,059-parameter design achieved 99.79%, and sharing its scalar MLP output bias with the attention triplet scalar preserves both learned biases and the load-bearing triplet/quartet attention boundary while adding only one cross-layer equality constraint.

INTENDED_EDIT: Adopt the verified scalar MLP bias and triplet/quartet attention bias, then reuse the attention triplet parameter as the MLP scalar bias, removing one deduplicated learned parameter.

EVIDENCE: The 1,059-parameter triplet/quartet attention design reached 99.79%; the failed quintet showed that the attention group boundary matters, while the 1,063-parameter scalar MLP bias independently reached 99.64%, motivating parameter sharing without deleting either mechanism.

<<<<<<< SEARCH
class TripletQuartetTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with a learned bias triplet and quartet."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 8:
            raise ValueError("out_features must be eight")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.free_bias[:1].expand(3), self.free_bias[1:].expand(4))
        )
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
        self.output_bias = nn.Parameter(torch.zeros(d_model - 2))
=======
        self.output_bias = nn.Parameter(torch.zeros(2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(2)))
=======
        bias = torch.cat(
            (self.output_bias[:1].expand(3), self.output_bias[1:].expand(4))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = TripletQuartetTiedBiasMeanZeroLinear(d_ff, d_model)
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.mlp = MLP(
            cfg.d_model, cfg.d_ff, cfg.dropout, self.attn.output_bias
        )
>>>>>>> REPLACE