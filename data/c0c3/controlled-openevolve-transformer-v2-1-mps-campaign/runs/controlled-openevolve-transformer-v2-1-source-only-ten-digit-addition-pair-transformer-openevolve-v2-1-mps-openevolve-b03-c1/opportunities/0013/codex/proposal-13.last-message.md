MECHANISM: Final-LayerNorm bias gauge fixing

HYPOTHESIS: Fixing one coordinate of the MLP output bias at zero will reduce the model from 1,604 to 1,603 parameters while retaining at least 99% accuracy, because any omitted bias vector is equivalent up to an all-coordinate shift erased by the final LayerNorm.

INTENDED_EDIT: Preserve the full 12-unit MLP and initialization ordering, but learn only seven `fc2` bias coordinates and pad the eighth with zero during the forward pass.

EVIDENCE: The 1,604-parameter design achieved 99.37%, while removing an entire feed-forward unit caused severe accuracy loss; this removes only a one-dimensional LayerNorm-invariant redundancy without narrowing the capacity-critical MLP.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # Preserve construction RNG, then fix one output-bias coordinate as the
        # representative of the common-shift equivalence erased by final LN.
        self.fc2.bias = nn.Parameter(self.fc2.bias.new_zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.fc2.bias, (0, 1))
        return self.drop(F.linear(F.gelu(self.fc1(x)), self.fc2.weight, bias))
>>>>>>> REPLACE