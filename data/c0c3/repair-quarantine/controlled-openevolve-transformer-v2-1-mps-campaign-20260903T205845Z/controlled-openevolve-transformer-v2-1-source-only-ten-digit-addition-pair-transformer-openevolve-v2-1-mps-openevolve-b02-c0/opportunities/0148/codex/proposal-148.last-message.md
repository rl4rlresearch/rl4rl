MECHANISM: Extend the attention-output residual common-mode gauge

HYPOTHESIS: Constraining a fifth attention projection column to have zero-mean outputs will reduce the model from 1,576 to 1,575 parameters while retaining at least 99% accuracy, because this removes an exact residual-stream common mode rather than another LayerNorm coordinate.

INTENDED_EDIT: Represent the first five, instead of four, attention-output weight columns in the existing zero-mean basis.

EVIDENCE: The 1,576-parameter model reached 99.92% with four attention columns already using this gauge, whereas removing another `ln1` bias coordinate reduced accuracy to 73.09%; this motivates extending the successful exact gauge instead of further restricting LayerNorm.

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