MECHANISM: Absorb pre-attention normalization scale into attention projections

HYPOTHESIS: Removing the eight learned `ln1` scale parameters while retaining a learned `ln1` bias will reduce the model from 1,620 to 1,612 parameters and preserve at least 99% accuracy.

INTENDED_EDIT: Make `ln1` non-affine, add a standalone learned bias, and apply that bias before causal self-attention.

EVIDENCE: Removing the absorbable `ln2` scale retained 99.96% accuracy, while removing `ln1` bias fell to 75.82%; this tests the analogous absorbable scale without discarding the optimization-critical bias.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.ln1_bias = nn.Parameter(torch.zeros(cfg.d_model))
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x) + self.ln1_bias)
        x = x + self.mlp(self.ln2(x))
>>>>>>> REPLACE