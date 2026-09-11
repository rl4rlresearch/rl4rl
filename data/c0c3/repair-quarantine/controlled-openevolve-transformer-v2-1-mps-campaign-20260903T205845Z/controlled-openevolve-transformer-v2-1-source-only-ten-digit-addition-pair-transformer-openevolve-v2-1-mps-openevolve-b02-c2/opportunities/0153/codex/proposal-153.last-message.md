MECHANISM: Incremental final-LayerNorm residual-column gauge fixing

HYPOTHESIS: Centering a fourth `fc2` output column will reduce the model from 1,581 to 1,580 learned parameters while retaining at least 99% accuracy; three centered columns achieved 0.9999, while the failure only appeared when all twelve were centered simultaneously.

INTENDED_EDIT: Extend `CompactResidualLinear` from three to four centered weight columns.

EVIDENCE: The verified three-column design reached 0.9999 accuracy with 1,581 parameters, whereas centering all twelve reduced accuracy to 0.9764; testing one additional column is the smallest evidence-grounded step toward the capacity boundary.

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