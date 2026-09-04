MECHANISM: Efficient mean-zero residual-output quotient

HYPOTHESIS: Constraining both residual branches to mean-zero outputs will reduce parameters from 1,384 to 1,373 while retaining at least 99% accuracy at 52,000 steps.

INTENDED_EDIT: Replace attention and MLP output projections with seven-coordinate mean-zero projections, fold the existing input basis into linear weights for faster execution, and reduce validation overhead.

EVIDENCE: The 1,384-parameter model achieved 99.99% accuracy; the prior broader quotient timed out, motivating isolation of its exact 11-parameter residual-output quotient with cheaper forward operations.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight @ self.basis.T
        return F.linear(x, weight, self.linear.bias)


class MeanZeroOutputLinear(nn.Module):
    """Linear map whose outputs lie in the mean-zero feature subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")
        self.linear = nn.Linear(in_features, out_features - 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.linear(x), (0, 1))
        return y - y.mean(dim=-1, keepdim=True)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=1000)
>>>>>>> REPLACE