MECHANISM: Gauge-aware anchored MLP output bias

HYPOTHESIS: Anchoring one coordinate of the shift-invariant MLP output bias and optimizing its seven relative coordinates with the proven implicit eight-coordinate AdamW state will reduce the model from 1,607 to 1,606 parameters while retaining at least 99% accuracy after 10,000 steps.

INTENDED_EDIT: Store seven `fc2` bias coordinates, reconstruct the eighth as zero, include this bias in gauge-aware optimization, and add 1,000 low-rate refinement steps.

EVIDENCE: Gauge-aware anchoring of the attention projection bias achieved 99.89% at 1,607 parameters, while the earlier learned cross-sublayer `fc2` bias tie failed at 49.09%; applying the successful optimizer-preserving gauge treatment avoids that harmful gradient coupling while exploiting the MLP bias’s equivalent uniform-shift invariance.

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc2_bias = torch.cat((self.fc2.bias, self.fc2.bias.new_zeros(1)))
        return self.drop(F.linear(F.gelu(self.fc1(x)), self.fc2.weight, fc2_bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [block.attn.proj.bias for block in model.blocks]
=======
    gauge_parameters = [
        param
        for block in model.blocks
        for param in (block.attn.proj.bias, block.mlp.fc2.bias)
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=9000)
=======
    p.add_argument("--train-steps", type=int, default=10000)
>>>>>>> REPLACE