MECHANISM: Extend sparse `fc1` row anchoring

HYPOTHESIS: Fixing the final input coefficient of a third `fc1` output row to zero will reduce the verified model from 1,570 to 1,569 learned parameters while retaining at least 99% accuracy after 21,000 training steps.

INTENDED_EDIT: Store three shortened `fc1` rows instead of two and reconstruct each row with a fixed zero final coefficient during the forward pass.

EVIDENCE: The current two-row anchoring design verified at 99.98% accuracy with 1,570 parameters, leaving substantial accuracy margin for testing one additional anchored coefficient.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.weight = nn.Parameter(torch.empty(d_ff * d_model - 2))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.weight = nn.Parameter(torch.empty(d_ff * d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        fc1_split = (
            (self.fc1.out_features - 2) * self.fc1.in_features
        )
        fc1_rows = self.fc1.weight[fc1_split:].view(
            2, self.fc1.in_features - 1
        )
        fc1_weight = torch.cat(
            (
                self.fc1.weight[:fc1_split],
                torch.cat(
                    (fc1_rows, fc1_rows.new_zeros(2, 1)), dim=1
                ).reshape(-1),
            )
        ).view(self.fc1.out_features, self.fc1.in_features)
=======
        fc1_split = (
            (self.fc1.out_features - 3) * self.fc1.in_features
        )
        fc1_rows = self.fc1.weight[fc1_split:].view(
            3, self.fc1.in_features - 1
        )
        fc1_weight = torch.cat(
            (
                self.fc1.weight[:fc1_split],
                torch.cat(
                    (fc1_rows, fc1_rows.new_zeros(3, 1)), dim=1
                ).reshape(-1),
            )
        ).view(self.fc1.out_features, self.fc1.in_features)
>>>>>>> REPLACE