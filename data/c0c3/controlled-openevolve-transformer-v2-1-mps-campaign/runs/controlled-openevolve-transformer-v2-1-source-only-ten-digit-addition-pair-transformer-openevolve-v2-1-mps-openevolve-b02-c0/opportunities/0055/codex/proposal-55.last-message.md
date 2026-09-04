MECHANISM: Fifth attention-output weight gauge elimination

HYPOTHESIS: Constraining the first five attention output-projection weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,616 to 1,615 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store and reconstruct the first five attention output-projection weight columns using independent orthonormal contrast coordinates, retaining the successful zero-mean output bias.

EVIDENCE: The identical incremental gauge retained 100%, 99.83%, 99.89%, and 99.94% accuracy with one through four constrained columns respectively; the latest result met the requirement at 1,616 parameters, directly motivating one further column.

<<<<<<< SEARCH
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
=======
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias and first five weight columns."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :5]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 5:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 5 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 5
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 5
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, self.bias_basis @ self.bias)
>>>>>>> REPLACE