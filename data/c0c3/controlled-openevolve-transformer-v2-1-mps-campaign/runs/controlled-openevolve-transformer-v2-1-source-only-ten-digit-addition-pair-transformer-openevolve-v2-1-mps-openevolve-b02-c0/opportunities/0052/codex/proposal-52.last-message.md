MECHANISM: Second attention-output weight gauge elimination

HYPOTHESIS: Constraining the first two attention output-projection weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,619 to 1,618 parameters while retaining at least 99% accuracy, because the first-column constraint achieved 100% accuracy and each removed component produces only a token-wise common-mode residual shift erased by downstream LayerNorms.

INTENDED_EDIT: Store and reconstruct the first two attention output-projection weight columns using independent orthonormal contrast coordinates, retaining the existing zero-mean output bias.

EVIDENCE: The immediately prior first-column orthonormal gauge reduction reached 100% accuracy at 1,619 parameters; extending the identical parameterization to one additional column is the closest controlled test of that successful mechanism.

<<<<<<< SEARCH
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias and first weight column."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        first_weight_coords = basis.transpose(0, 1) @ full_weight[:, 0]
        self.weight = nn.Parameter(
            torch.cat((first_weight_coords, full_weight[:, 1:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_weight_column = self.bias_basis @ self.weight[: self.out_features - 1]
        remaining_weight = self.weight[self.out_features - 1 :].view(
            self.out_features, self.in_features - 1
        )
        weight = torch.cat((first_weight_column.unsqueeze(1), remaining_weight), dim=1)
        return F.linear(x, weight, self.bias_basis @ self.bias)
=======
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias and first two weight columns."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :2]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 2:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 2 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 2
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 2
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, self.bias_basis @ self.bias)
>>>>>>> REPLACE