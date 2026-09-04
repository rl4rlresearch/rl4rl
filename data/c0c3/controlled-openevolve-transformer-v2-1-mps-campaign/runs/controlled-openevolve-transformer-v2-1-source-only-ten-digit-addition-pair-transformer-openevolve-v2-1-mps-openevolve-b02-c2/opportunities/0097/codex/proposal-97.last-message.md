MECHANISM: Activation-scaled residual-uniform MLP output gauge quotient

HYPOTHESIS: Helmert-parameterizing one `fc2` weight column will reduce the verified 1,595-parameter model to 1,594 parameters while retaining at least 99% accuracy, because the removed column mean contributes only a per-token uniform residual shift eliminated by downstream LayerNorm.

INTENDED_EDIT: Extend `CompactResidualLinear` to reconstruct its first weight column from seven zero-mean coordinates while retaining the other columns and the qualified compact bias.

EVIDENCE: The current 1,595-parameter design achieved 99.97% accuracy and already removes the residual-uniform direction from `fc2` bias; applying the same exact output-space quotient to one activation-scaled weight column is more directly supported than repeating the second `fc1` row compaction that fell to 37.28%.

<<<<<<< SEARCH
class CompactResidualLinear(nn.Module):
    """Linear layer with its residual-uniform bias direction fixed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_bias = linear.bias - linear.bias.mean()
        self.bias = nn.Parameter(
            (basis.transpose(0, 1) @ centered_bias).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, self.weight, full_bias)
=======
class CompactResidualLinear(nn.Module):
    """Linear layer with one weight-column and bias uniform directions fixed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_column = linear.weight[:, 0] - linear.weight[:, 0].mean()
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_column).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 1:].detach().clone())

        centered_bias = linear.bias - linear.bias.mean()
        self.bias = nn.Parameter(
            (basis.transpose(0, 1) @ centered_bias).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_column = self.bias_basis @ self.column_weight
        full_weight = torch.cat(
            (first_column.unsqueeze(1), self.weight),
            dim=1,
        )
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE