MECHANISM: Three-row fc1 anchoring with reduced validation overhead

HYPOTHESIS: Anchoring a third `fc1` coefficient will reduce the model to 1,569 parameters while preserving at least 99% accuracy, and less-frequent evaluation will allow all 21,000 training steps to finish within the verification limit.

INTENDED_EDIT: Store three shortened `fc1` rows with fixed zero final coefficients and change the default evaluation interval from 300 to 1,000 steps.

EVIDENCE: The two-row design reached 99.98% accuracy with 1,570 parameters; the prior three-row attempt failed only because training timed out, so reducing evaluation overhead directly addresses the observed failure while retaining the tested parameter reduction.

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

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=1000)
>>>>>>> REPLACE