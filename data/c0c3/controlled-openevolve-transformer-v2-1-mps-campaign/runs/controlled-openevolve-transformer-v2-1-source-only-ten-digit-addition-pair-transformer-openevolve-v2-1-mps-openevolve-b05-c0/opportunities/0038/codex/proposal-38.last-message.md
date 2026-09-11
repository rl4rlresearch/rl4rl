MECHANISM: Post-MLP residual common-shift quotient

HYPOTHESIS: Fixing one coordinate of the MLP output bias will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy, because an all-coordinate bias shift survives only in the residual stream and is removed by subsequent LayerNorms.

INTENDED_EDIT: Store seven MLP output-bias coordinates, reconstruct the eighth as zero during the learned projection, and train the reduced bias with the existing virtual-coordinate AdamW and gradient-clipping logic.

EVIDENCE: The 1622-parameter design reached 99.92%, while additional first-head key gauges repeatedly failed and the remaining value-bias coordinate was historically the weakest removal candidate; the current successful attention projection-bias quotient already uses the same LayerNorm-null common-shift mechanism and optimizer treatment.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        output = F.linear(
            hidden,
            self.fc2.weight,
            F.pad(self.fc2.bias, (0, 1)),
        )
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
=======
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
        *[block.mlp.fc2.bias for block in model.blocks],
    ]
>>>>>>> REPLACE