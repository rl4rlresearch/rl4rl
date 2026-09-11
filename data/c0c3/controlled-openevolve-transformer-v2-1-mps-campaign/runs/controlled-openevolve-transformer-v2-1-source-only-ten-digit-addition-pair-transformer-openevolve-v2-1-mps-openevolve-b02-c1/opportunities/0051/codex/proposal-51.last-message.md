MECHANISM: Complementary nonlinear-branch isolation

HYPOTHESIS: Removing the seven-parameter MLP residual bias will reduce the model from 1,053 to 1,046 parameters while retaining at least 99% accuracy, because the width-1 model achieved 100% and the failed bias-only limit indicates the nonlinear neuron—not the standalone bias—is the load-bearing component.

INTENDED_EDIT: Delete the learned residual bias from the final one-neuron MLP while preserving its learned GELU input and output projections.

EVIDENCE: The width-1 MLP reached 100% accuracy, whereas retaining only its residual bias collapsed accuracy to 23.29%; the complementary ablation directly tests whether that bias contributes anything when the necessary nonlinear pathway remains intact.

<<<<<<< SEARCH
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.fc2(F.gelu(self.fc1(x[..., :-1]))), (0, 1))
        y = y + F.pad(self.fc2_bias, (0, 1))
        return self.drop(y)
=======
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.fc2(F.gelu(self.fc1(x[..., :-1]))), (0, 1))
        return self.drop(y)
>>>>>>> REPLACE