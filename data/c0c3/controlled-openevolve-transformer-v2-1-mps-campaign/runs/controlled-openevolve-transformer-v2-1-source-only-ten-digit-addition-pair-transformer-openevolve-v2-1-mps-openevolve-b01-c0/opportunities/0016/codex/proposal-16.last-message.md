MECHANISM: Terminal LayerNorm-null MLP output-bias gauge

HYPOTHESIS: Constraining `fc2` bias to seven learned zero-mean contrasts will reduce the model to 1631 parameters while retaining at least 99% accuracy, because its eliminated common-offset direction is exactly removed by the immediately following final LayerNorm.

INTENDED_EDIT: Preserve `fc2` construction and initialization while replacing its eight-element bias with seven learned contrasts that reconstruct a centered eight-coordinate bias.

EVIDENCE: Balanced bias contrasts in `ln1` and `ln2` retained 99.93% and 99.96% accuracy; unlike the failed attention projection-bias reduction, `fc2` is terminal and its common bias shift reaches final LayerNorm without an intervening computation.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.fc2.bias = None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        output_bias = torch.cat((self.fc2_bias, self.fc2_bias.new_zeros(1)))
        output_bias = output_bias - output_bias.mean()
        return self.drop(F.linear(hidden, self.fc2.weight, output_bias))
>>>>>>> REPLACE