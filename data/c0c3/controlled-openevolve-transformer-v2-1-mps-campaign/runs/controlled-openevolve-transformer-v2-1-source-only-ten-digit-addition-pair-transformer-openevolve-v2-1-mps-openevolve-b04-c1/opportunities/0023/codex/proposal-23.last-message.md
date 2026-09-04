MECHANISM: Final-residual MLP output-bias gauge fixing

HYPOTHESIS: Fixing one MLP output-bias coordinate to zero will reduce the model from 1608 to 1607 parameters while retaining at least 99% accuracy, because the final LayerNorm erases the all-ones component of this residual-branch bias.

INTENDED_EDIT: Replace the eight-coordinate second MLP layer bias with seven learned coordinates padded by one fixed zero.

EVIDENCE: The analogous attention output-bias gauge fixing reduced the model from 1609 to 1608 parameters while achieving 99.88% accuracy; the MLP output is likewise added to the residual stream immediately before a LayerNorm.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        return self.drop(self.fc2(F.gelu(self.fc1(x))) + F.pad(self.fc2_bias, (0, 1)))
>>>>>>> REPLACE