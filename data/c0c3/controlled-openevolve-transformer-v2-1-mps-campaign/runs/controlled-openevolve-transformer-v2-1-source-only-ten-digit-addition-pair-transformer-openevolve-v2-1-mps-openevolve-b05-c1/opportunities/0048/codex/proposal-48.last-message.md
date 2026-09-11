MECHANISM: Shared MLP bias coordinate

HYPOTHESIS: Reusing the mean of the seven-coordinate MLP output bias as the hidden-unit bias will reduce the model from 1,374 to 1,373 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Remove the standalone hidden-bias parameter and derive its zero-initialized value from `fc2.bias`.

EVIDENCE: The 1,375-parameter model achieved 99.99% accuracy while deriving its hidden bias from the mean of `fc2.bias`; the current 1,374-parameter model achieved 100%, so this directly tests whether that successful sharing remains sufficient after quotienting the MLP output weight.

<<<<<<< SEARCH
        self.fc1 = MeanZeroInputLinear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model - 1)
        self.hidden_bias = nn.Parameter(torch.zeros(()))
        self.drop = nn.Dropout(dropout)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model - 1)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = self.fc1(x) + self.hidden_bias
=======
        hidden = self.fc1(x) + self.fc2.bias.mean()
>>>>>>> REPLACE