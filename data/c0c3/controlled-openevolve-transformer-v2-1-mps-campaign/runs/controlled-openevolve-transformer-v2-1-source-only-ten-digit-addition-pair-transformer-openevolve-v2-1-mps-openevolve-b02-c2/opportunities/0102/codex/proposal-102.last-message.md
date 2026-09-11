MECHANISM: Fourth downstream-affine LayerNorm-bias gauge

HYPOTHESIS: Reproducing the verified 1,593-parameter two-column `fc2` design and fixing a fourth `ln2` bias coordinate will yield 1,592 parameters with at least 99% accuracy, because the retained independent `fc1` biases absorb that LayerNorm offset without restricting the learned function.

INTENDED_EDIT: Reconstruct the first two `fc2` columns from zero-mean coordinates and increase `ln2`’s fixed bias coordinates from three to four.

EVIDENCE: The two-column `fc2` quotient achieved 99.96% accuracy at 1,593 parameters. Unlike the failed third-`fc2` column, attention-projection column, and Q/K rotation, this tests an independent downstream-affine bias redundancy while preserving the qualified `fc2` layout.

<<<<<<< SEARCH
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
=======
class CompactResidualLinear(nn.Module):
    """Linear layer with two weight-column and bias uniform directions fixed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_columns = (
            linear.weight[:, :2]
            - linear.weight[:, :2].mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 2:].detach().clone())

        centered_bias = linear.bias - linear.bias.mean()
        self.bias = nn.Parameter(
            (basis.transpose(0, 1) @ centered_bias).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compact_columns = self.bias_basis @ self.column_weight
        full_weight = torch.cat((compact_columns, self.weight), dim=1)
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=3,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=4,
        )
>>>>>>> REPLACE