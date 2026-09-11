MECHANISM: Final-LayerNorm quotient for MLP output bias

HYPOTHESIS: Representing the `fc2` bias with seven orthonormal mean-free coordinates will reduce parameters from 1,618 to 1,617 while retaining at least 99% accuracy, because its omitted all-ones component is canceled by the downstream final LayerNorm.

INTENDED_EDIT: Replace the eight-parameter MLP output bias with a seven-parameter basis representation, preserving constructor RNG consumption, MLP width, and all observable bias directions.

EVIDENCE: Mean-free positional embeddings removed the same LayerNorm-invariant direction at every position and achieved 99.64% accuracy; the current design subsequently achieved 99.98%, supporting another exact one-parameter LayerNorm quotient instead of reducing load-bearing capacity.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class MeanFreeResidualLinear(nn.Linear):
    """Linear output bias modulo the constant direction removed by LayerNorm."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

        basis = torch.zeros(out_features, out_features - 1)
        for j in range(out_features - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = self.bias @ self.bias_basis.transpose(0, 1)
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = MeanFreeResidualLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE