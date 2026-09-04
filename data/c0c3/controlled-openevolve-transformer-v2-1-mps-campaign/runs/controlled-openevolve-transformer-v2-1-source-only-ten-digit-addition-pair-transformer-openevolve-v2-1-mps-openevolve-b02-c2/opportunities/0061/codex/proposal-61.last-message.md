MECHANISM: Orthonormal attention-projection residual gauge quotient

HYPOTHESIS: Expressing one attention output-projection column in a seven-dimensional zero-mean basis will reduce the model to 1,617 parameters while retaining at least 99% accuracy, because the removed component produces only a token-dependent uniform residual shift canceled by `ln2` and `ln_f`.

INTENDED_EDIT: Reparameterize the first attention output-projection column with a Helmert basis while preserving the existing shared value/projection offset quotient.

EVIDENCE: The current 1,618-parameter model reached 99.96%, and the orthonormal joint value/projection common-mode quotient reached 99.57%; the failed fourth LayerNorm-coupled key quotient motivates testing the corresponding fixed-basis residual invariance instead.

<<<<<<< SEARCH
class CompactSharedProjection(nn.Module):
    """Projection with a zero-mean effective offset and retained value scalar."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = linear.weight
        self.shared_bias = shared_bias

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 2)
        for column in range(width - 2):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_bias = linear.bias - linear.bias.mean()
        compact_bias = basis.transpose(0, 1) @ centered_bias
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_bias = self.bias_basis @ self.bias
        raw_bias = torch.cat(
            (
                raw_bias[:-1],
                raw_bias[-1:] + self.shared_bias,
            )
        )
        value_offset = self.weight[:, -1] * self.shared_bias
        full_bias = raw_bias - (raw_bias + value_offset).mean()
        return F.linear(x, self.weight, full_bias)
=======
class CompactSharedProjection(nn.Module):
    """Projection with zero-mean weight and effective-offset gauges fixed."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.shared_bias = shared_bias

        width = linear.out_features
        weight_basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            weight_basis[: column + 1, column] = 1.0 / denom
            weight_basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("weight_basis", weight_basis, persistent=False)

        centered_weight_column = linear.weight[:, 0] - linear.weight[:, 0].mean()
        self.weight_column = nn.Parameter(
            (weight_basis.transpose(0, 1) @ centered_weight_column).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 1:].detach().clone())

        bias_basis = linear.weight.new_zeros(width, width - 2)
        for column in range(width - 2):
            denom = math.sqrt((column + 1) * (column + 2))
            bias_basis[: column + 1, column] = 1.0 / denom
            bias_basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", bias_basis, persistent=False)

        centered_bias = linear.bias - linear.bias.mean()
        compact_bias = bias_basis.transpose(0, 1) @ centered_bias
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_column = self.weight_basis @ self.weight_column
        full_weight = torch.cat((weight_column.unsqueeze(1), self.weight), dim=1)

        raw_bias = self.bias_basis @ self.bias
        raw_bias = torch.cat(
            (
                raw_bias[:-1],
                raw_bias[-1:] + self.shared_bias,
            )
        )
        value_offset = full_weight[:, -1] * self.shared_bias
        full_bias = raw_bias - (raw_bias + value_offset).mean()
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE