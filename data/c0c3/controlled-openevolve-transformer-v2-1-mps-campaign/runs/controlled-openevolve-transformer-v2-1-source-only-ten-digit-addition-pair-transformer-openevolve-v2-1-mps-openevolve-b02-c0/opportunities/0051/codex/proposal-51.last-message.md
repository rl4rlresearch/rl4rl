MECHANISM: Orthonormal attention-output weight gauge elimination

HYPOTHESIS: Constraining the first attention output-projection weight column to its seven-dimensional zero-mean subspace will reduce the model from 1,620 to 1,619 parameters while retaining at least 99% accuracy, because the removed component produces only token-wise common-mode residual shifts erased by subsequent LayerNorms.

INTENDED_EDIT: Store the first attention output-projection weight column as seven orthonormal contrast coordinates and reconstruct the full zero-mean column during forward passes, while retaining the successful zero-mean output bias.

EVIDENCE: The orthonormal attention-output bias gauge retained 99.89% accuracy at 1,625 parameters, demonstrating that this projection tolerates orthonormal common-mode quotienting; this applies the same successful parameterization to one weight column rather than extending the position-specific gauges that failed at 1,619 parameters.

<<<<<<< SEARCH
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias_basis @ self.bias)
=======
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
>>>>>>> REPLACE