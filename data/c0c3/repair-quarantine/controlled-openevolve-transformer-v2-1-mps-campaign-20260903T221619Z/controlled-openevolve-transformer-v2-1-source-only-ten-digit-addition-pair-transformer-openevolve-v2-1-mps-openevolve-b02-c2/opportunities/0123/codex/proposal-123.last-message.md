MECHANISM: Fourth MLP residual-output uniform gauge quotient

HYPOTHESIS: Centering the fourth `fc2` output column will reduce the verified 1,581-parameter model to 1,580 parameters while retaining at least 99% accuracy, because its uniform residual component is removed exactly by the final LayerNorm.

INTENDED_EDIT: Extend `CompactResidualLinear` from three to four centered weight columns while preserving all other qualified compactions and training settings.

EVIDENCE: Centering the third `fc2` column produced 99.87% accuracy at 1,581 parameters; the fourth column has the identical final-LayerNorm-null gauge structure, making it the closest supported one-parameter reduction.

<<<<<<< SEARCH
class CompactResidualLinear(nn.Module):
    """Linear layer with three weight-column and bias uniform directions fixed."""

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
            linear.weight[:, :3] - linear.weight[:, :3].mean(
                dim=0,
                keepdim=True,
            )
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 3:].detach().clone())
=======
class CompactResidualLinear(nn.Module):
    """Linear layer with four weight-column and bias uniform directions fixed."""

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
            linear.weight[:, :4] - linear.weight[:, :4].mean(
                dim=0,
                keepdim=True,
            )
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 4:].detach().clone())
>>>>>>> REPLACE