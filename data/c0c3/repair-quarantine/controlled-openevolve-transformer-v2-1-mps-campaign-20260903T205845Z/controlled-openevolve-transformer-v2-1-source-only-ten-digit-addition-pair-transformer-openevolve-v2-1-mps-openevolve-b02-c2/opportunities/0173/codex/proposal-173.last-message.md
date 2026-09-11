MECHANISM: Incremental residual-output gauge fixing

HYPOTHESIS: Centering a sixth `fc2` output column will reduce the verified model from 1,579 to 1,578 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Extend `CompactResidualLinear` from five to six zero-mean output columns along the uniform residual direction.

EVIDENCE: The current five-column design achieved 0.998 accuracy with 1,579 parameters, while four columns achieved 0.9974; testing the adjacent sixth column is the smallest informative reduction before the known failure at twelve columns.

<<<<<<< SEARCH
class CompactResidualLinear(nn.Module):
    """Linear layer with five weight-column and bias uniform directions fixed."""

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
            linear.weight[:, :5]
            - linear.weight[:, :5].mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 5:].detach().clone())
=======
class CompactResidualLinear(nn.Module):
    """Linear layer with six weight-column and bias uniform directions fixed."""

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
            linear.weight[:, :6]
            - linear.weight[:, :6].mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 6:].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        # also quotient one independently biased MLP input row and five MLP
        # output-column uniform directions.
=======
        # also quotient one independently biased MLP input row and six MLP
        # output-column uniform directions.
>>>>>>> REPLACE