MECHANISM: Single-scalar MLP synthesis constraint

HYPOTHESIS: Fixing one trailing MLP output-projection coefficient at zero will reduce the qualified model to 1,137 learned parameters while retaining at least 99% accuracy after 45,000 updates.

INTENDED_EDIT: Learn 95 of the 96 MLP synthesis weights, reconstruct the final coefficient as zero, and preserve full constructor initialization RNG consumption.

EVIDENCE: The 1,138-parameter model achieved 99.78% accuracy, while further attention-output constraints failed sharply; the MLP remains intact, and a one-scalar constraint is the smallest informative probe of redundancy in that orthogonal subsystem.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # Fix one trailing synthesis coefficient while preserving the full
        # Linear constructor's initialization RNG consumption.
        self.fc2.weight = nn.Parameter(torch.empty(d_model * d_ff - 1))
        self.fc2.fixed_weight_trim = (0, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        fc2_weight = F.pad(self.fc2.weight, (0, 1)).view(
            self.fc2.out_features,
            self.fc2.in_features,
        )
        return self.drop(F.linear(hidden, fc2_weight, self.fc2.bias))
>>>>>>> REPLACE