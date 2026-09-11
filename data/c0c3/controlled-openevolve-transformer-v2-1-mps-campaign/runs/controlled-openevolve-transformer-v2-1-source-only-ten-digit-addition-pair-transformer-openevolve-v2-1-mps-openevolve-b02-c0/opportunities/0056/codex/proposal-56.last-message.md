MECHANISM: MLP-output common-mode gauge elimination

HYPOTHESIS: Constraining the first MLP output-projection weight column to its seven-dimensional zero-mean subspace will reduce the model from 1,616 to 1,615 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store the first `fc2` weight column as orthonormal contrast coordinates and reconstruct it during forward passes, while preserving the existing anchored output bias.

EVIDENCE: Four attention output-projection columns accepted the identical common-mode quotient with 99.94% accuracy at 1,616 parameters; applying it to the separate MLP output projection tests the same successful invariance without extending into the fifth attention column that collapsed to 73.99%.

<<<<<<< SEARCH
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with the common-mode output bias gauge fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, F.pad(self.bias, (0, 1)))
=======
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
>>>>>>> REPLACE