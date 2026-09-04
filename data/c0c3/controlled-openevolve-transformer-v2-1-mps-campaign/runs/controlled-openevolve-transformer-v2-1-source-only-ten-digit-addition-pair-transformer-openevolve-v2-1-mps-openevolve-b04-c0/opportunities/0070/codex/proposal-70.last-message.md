MECHANISM: Final-residual bias gauge anchoring

HYPOTHESIS: Fixing `fc2.bias[0]` at zero will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy, because any removed value can be subtracted uniformly from all eight bias coordinates without changing the following final LayerNorm output.

INTENDED_EDIT: Replace the eight-parameter `fc2` bias with seven learned coordinates, reconstruct coordinate 0 as zero, and preserve ordinary `nn.Linear` initialization RNG consumption.

EVIDENCE: The 1593-parameter design achieved 99.90%, while every attempted fourth query-bias removal failed; this motivates testing an untouched exact gauge in the final MLP residual rather than further reducing query capacity.

<<<<<<< SEARCH
        self.fc1 = BiasAnchoredLinear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = BiasAnchoredLinear(d_model, d_ff)
        # Construct the ordinary bias first to preserve initialization RNG use.
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.bias = None
        self.fc2_bias_rest = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        zero = self.fc2_bias_rest.new_zeros(1)
        bias = torch.cat((zero, self.fc2_bias_rest))
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
>>>>>>> REPLACE