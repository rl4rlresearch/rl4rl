MECHANISM: Quotient-aware MLP output-bias gauge fixing

HYPOTHESIS: Removing the LayerNorm-invariant common-shift coordinate from the MLP output bias will reduce the model from 1634 to 1633 parameters while retaining at least 99% accuracy when virtual eight-coordinate AdamW preserves its optimizer dynamics.

INTENDED_EDIT: Replace the eight-coordinate MLP output bias with seven learned coordinates and one fixed zero, then include it in the existing gauge-aware optimizer and gradient clipping.

EVIDENCE: Virtual full-coordinate AdamW made the analogous attention output-bias reduction reach 99.85% at 1634 parameters; unlike the failed positional gauge, the MLP bias is another zero-initialized output offset whose common shift passes through the residual stream and is canceled by downstream LayerNorms.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        return self.drop(F.linear(hidden, self.fc2.weight, F.pad(self.fc2_bias, (0, 1))))
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [block.attn.proj_bias for block in model.blocks]
=======
    gauge_parameters = [
        parameter
        for block in model.blocks
        for parameter in (block.attn.proj_bias, block.mlp.fc2_bias)
    ]
>>>>>>> REPLACE