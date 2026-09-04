MECHANISM: Fourth attention-output weight gauge elimination

HYPOTHESIS: Constraining the first four attention output-projection weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,617 to 1,616 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store and reconstruct the first four attention output-projection weight columns using independent orthonormal contrast coordinates, retaining the successful zero-mean output bias.

EVIDENCE: The same incremental gauge reduction retained 100% accuracy with one constrained column, 99.83% with two, and 99.89% with three at 1,617 parameters; extending it by one column is the closest controlled next test.

<<<<<<< SEARCH
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias and first three weight columns."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :3]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 3:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 3 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 3
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, self.bias_basis @ self.bias)
=======
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias and first four weight columns."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :4]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 4:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 4 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 4
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 4
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, self.bias_basis @ self.bias)
>>>>>>> REPLACE