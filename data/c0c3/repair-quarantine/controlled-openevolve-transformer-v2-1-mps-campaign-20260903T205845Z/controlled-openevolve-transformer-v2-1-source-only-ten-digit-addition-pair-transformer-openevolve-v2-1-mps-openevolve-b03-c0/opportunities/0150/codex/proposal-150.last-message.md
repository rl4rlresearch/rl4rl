MECHANISM: Third-row MLP input-coordinate anchoring

HYPOTHESIS: Fixing the final input coefficient of a third `fc1` output row at zero will reduce the model from 1,570 to 1,569 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store three shortened `fc1` rows and reconstruct each with a zero final coefficient during the forward pass.

EVIDENCE: The current two-row anchoring design achieved 0.9998 accuracy with 1,570 parameters, leaving substantial accuracy margin for the next one-parameter ablation.

<<<<<<< SEARCH
        self.fc1.weight = nn.Parameter(torch.empty(d_ff * d_model - 2))
=======
        self.fc1.weight = nn.Parameter(torch.empty(d_ff * d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        fc1_split = (
            (self.fc1.out_features - 2) * self.fc1.in_features
        )
        fc1_rows = self.fc1.weight[fc1_split:].view(
            2, self.fc1.in_features - 1
        )
=======
        fc1_split = (
            (self.fc1.out_features - 3) * self.fc1.in_features
        )
        fc1_rows = self.fc1.weight[fc1_split:].view(
            3, self.fc1.in_features - 1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                    (fc1_rows, fc1_rows.new_zeros(2, 1)), dim=1
=======
                    (fc1_rows, fc1_rows.new_zeros(3, 1)), dim=1
>>>>>>> REPLACE