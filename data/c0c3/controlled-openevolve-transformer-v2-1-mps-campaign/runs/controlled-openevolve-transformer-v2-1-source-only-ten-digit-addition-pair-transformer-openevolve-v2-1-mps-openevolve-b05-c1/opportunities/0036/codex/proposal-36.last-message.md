MECHANISM: Zero-copy mean-zero coordinates with bias-free terminal normalization

HYPOTHESIS: Replacing the two fixed-basis matrix multiplications with zero-copy coordinate slices will provide enough runtime margin for the 52,000-step bias-free model to finish, reducing parameters from 1,384 to 1,376 while retaining at least 99% accuracy.

INTENDED_EDIT: Use the first seven coordinates as a complete chart for LayerNorm’s mean-zero outputs, disable the final LayerNorm bias, and evaluate only at initialization and the final step.

EVIDENCE: The 1,384-parameter model achieved 99.99% at 52,000 steps, while every 1,376-parameter bias-free attempt failed only by timeout. The current orthogonal quotient performs two additional basis matrix multiplications per forward pass, so an equivalent zero-copy parameterization plus minimal validation directly addresses the observed failure mode.

<<<<<<< SEARCH
class MeanZeroInputLinear(nn.Module):
    """Linear map restricted to the mean-zero input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")

        basis = torch.zeros(in_features, in_features - 1)
        for j in range(in_features - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
        self.linear = nn.Linear(in_features - 1, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)
=======
class MeanZeroInputLinear(nn.Module):
    """Linear map restricted to the mean-zero input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")
        self.linear = nn.Linear(in_features - 1, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm guarantees that the omitted coordinate is determined by
        # the retained coordinates, so this is a complete zero-copy chart.
        return self.linear(x[..., :-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=52000)
>>>>>>> REPLACE