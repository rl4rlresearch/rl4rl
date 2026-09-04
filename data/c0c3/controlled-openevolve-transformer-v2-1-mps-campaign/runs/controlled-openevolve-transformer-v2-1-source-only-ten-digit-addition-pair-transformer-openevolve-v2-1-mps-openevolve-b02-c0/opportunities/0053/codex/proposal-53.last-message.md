MECHANISM: Third attention-output weight gauge elimination

HYPOTHESIS: Constraining the first three attention output-projection weight columns independently to seven-dimensional zero-mean subspaces will reduce the model from 1,618 to 1,617 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store and reconstruct the first three attention output-projection weight columns using independent orthonormal contrast coordinates, retaining the successful zero-mean output bias.

EVIDENCE: Extending this exact orthonormal gauge from one column at 100% accuracy and 1,619 parameters to two columns retained 99.83% accuracy at 1,618 parameters, directly motivating one further incremental extension.

<<<<<<< SEARCH
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias and first two weight columns."""
=======
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias and first three weight columns."""
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
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 2
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 2
        )
=======
        gauge_size = 3 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 3
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 3
        )
>>>>>>> REPLACE