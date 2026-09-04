MECHANISM: Trailing attention-projection residual-uniform gauge quotient

HYPOTHESIS: Centering the final attention output-projection column will yield 1,579 parameters with at least 99% accuracy, because its removed uniform residual component is canceled by subsequent LayerNorms, while the trailing-coordinate layout may avoid the optimization weakness observed when centering the first column.

INTENDED_EDIT: Represent the final `attn.proj` input column in a seven-dimensional zero-mean basis and reconstruct it during the forward pass.

EVIDENCE: The 1,580-parameter design achieved 99.30%; centering the first attention-projection column narrowly missed at 98.91%, while the analogous trailing, nonadjacent `fc2` column succeeded where an adjacent column failed, motivating a trailing-coordinate test of the same exact gauge.

<<<<<<< SEARCH
class CompactSharedProjection(nn.Module):
    """Projection with a zero-mean effective offset and retained value scalar."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = linear.weight
        self.shared_bias = shared_bias
=======
class CompactSharedProjection(nn.Module):
    """Projection with centered final column, offset, and retained value scalar."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = nn.Parameter(
            linear.weight[:, :-1].detach().clone()
        )
        self.shared_bias = shared_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_bias = linear.bias - linear.bias.mean()
=======
        self.register_buffer("bias_basis", basis, persistent=False)

        column_basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            column_basis[: column + 1, column] = 1.0 / denom
            column_basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("column_basis", column_basis, persistent=False)

        centered_column = (
            linear.weight[:, -1] - linear.weight[:, -1].mean()
        )
        self.column_weight = nn.Parameter(
            (column_basis.transpose(0, 1) @ centered_column).detach().clone()
        )

        centered_bias = linear.bias - linear.bias.mean()
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_offset = self.weight[:, -1] * self.shared_bias
        full_bias = raw_bias - (raw_bias + value_offset).mean()
        return F.linear(x, self.weight, full_bias)
=======
        final_column = self.column_basis @ self.column_weight
        full_weight = torch.cat(
            (self.weight, final_column.unsqueeze(1)),
            dim=1,
        )
        value_offset = final_column * self.shared_bias
        full_bias = raw_bias - (raw_bias + value_offset).mean()
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE