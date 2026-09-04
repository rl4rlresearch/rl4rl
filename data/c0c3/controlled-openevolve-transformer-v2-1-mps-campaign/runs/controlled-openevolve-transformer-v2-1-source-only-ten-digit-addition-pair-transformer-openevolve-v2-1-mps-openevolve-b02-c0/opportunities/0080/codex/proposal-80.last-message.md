MECHANISM: Incremental terminal-MLP common-mode gauge fixing

HYPOTHESIS: Constraining a second `fc2` weight column to have zero mean will reduce the model from 1,585 to 1,584 parameters while retaining at least 99% accuracy, because its removed common-mode output is canceled by downstream LayerNorms.

INTENDED_EDIT: Extend `OutputAnchoredLinear`’s orthogonal zero-mean parameterization from its first weight column to its first two columns.

EVIDENCE: The verified 1,585-parameter design achieved 99.93% accuracy while already constraining the first `fc2` column; the failed fifth attention-column constraint motivates testing the next independent downstream gauge without further restricting attention.

<<<<<<< SEARCH
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and a zero-mean first weight column."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        basis = self.weight.detach().new_zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("weight_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        first_weight_coords = basis.transpose(0, 1) @ full_weight[:, :1]
        self.weight = nn.Parameter(
            torch.cat((first_weight_coords.flatten(), full_weight[:, 1:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_weight = self.weight_basis @ self.weight[: self.out_features - 1]
        remaining_weight = self.weight[self.out_features - 1 :].view(
            self.out_features, self.in_features - 1
        )
        weight = torch.cat((first_weight.unsqueeze(1), remaining_weight), dim=1)
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
=======
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and two zero-mean weight columns."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        basis = self.weight.detach().new_zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("weight_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :2]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 2:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 2 * (self.out_features - 1)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 2
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 2
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
>>>>>>> REPLACE