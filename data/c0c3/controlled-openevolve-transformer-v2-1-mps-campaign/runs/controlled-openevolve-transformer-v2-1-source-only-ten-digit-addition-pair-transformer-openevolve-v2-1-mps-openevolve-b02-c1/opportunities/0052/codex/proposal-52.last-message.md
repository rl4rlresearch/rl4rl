MECHANISM: Bias-free nonlinear bottleneck

HYPOTHESIS: Removing the final MLP neuron’s scalar input bias will reduce the model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because the nonlinear pathway is necessary but the latest successful result shows its seven-parameter residual bias is not.

INTENDED_EDIT: Make the width-one MLP input projection bias-free while consuming the removed constructor draw so every remaining parameter keeps the current initialization stream.

EVIDENCE: The width-one MLP achieved 100% accuracy, the bias-only replacement collapsed to 23.29%, and removing the nonlinear branch’s residual bias still achieved 99.95%; this isolates the only remaining scalar bias in the load-bearing nonlinear unit with the smallest possible ablation.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
=======
        self.fc1 = nn.Linear(d_model - 1, d_ff, bias=False)

        # Preserve the RNG stream of the removed fc1 bias initialization.
        fc1_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(d_ff).uniform_(-fc1_bound, fc1_bound)

        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
>>>>>>> REPLACE