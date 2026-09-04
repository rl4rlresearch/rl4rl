MECHANISM: Second MLP-output common-mode gauge elimination

HYPOTHESIS: Constraining the first two `fc2` weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,615 to 1,614 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store and reconstruct the first two MLP output-projection weight columns using independent orthonormal contrast coordinates while preserving the anchored output bias.

EVIDENCE: Constraining the first `fc2` column retained 99.93% accuracy at 1,615 parameters; extending the identical parameterization by one column is the closest controlled test of that successful mechanism.

<<<<<<< SEARCH
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and a zero-mean first weight column."""
=======
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and first two zero-mean weight columns."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_weight = self.weight.detach()
        first_weight_coords = basis.transpose(0, 1) @ full_weight[:, :1]
        self.weight = nn.Parameter(
            torch.cat((first_weight_coords.flatten(), full_weight[:, 1:].flatten()))
        )
=======
        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :2]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 2:].flatten()))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_weight = self.weight_basis @ self.weight[: self.out_features - 1]
        remaining_weight = self.weight[self.out_features - 1 :].view(
            self.out_features, self.in_features - 1
        )
        weight = torch.cat((first_weight.unsqueeze(1), remaining_weight), dim=1)
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
=======
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