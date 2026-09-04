MECHANISM: Incremental terminal-MLP common-mode gauge fixing

HYPOTHESIS: Constraining a third `fc2` weight column to have zero mean will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because downstream LayerNorm removes the eliminated common-mode output.

INTENDED_EDIT: Extend `OutputAnchoredLinear`’s zero-mean parameterization from its first two weight columns to its first three.

EVIDENCE: The immediately preceding identical extension from one to two constrained `fc2` columns achieved 99.97% accuracy at 1,584 parameters, making a third column the closest controlled reduction.

<<<<<<< SEARCH
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and two zero-mean weight columns."""
=======
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and three zero-mean weight columns."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :2]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 2:].flatten()))
        )
=======
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :3]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 3:].flatten()))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauge_size = 2 * (self.out_features - 1)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 2
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 2
        )
=======
        gauge_size = 3 * (self.out_features - 1)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 3
        )
>>>>>>> REPLACE