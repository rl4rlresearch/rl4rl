MECHANISM: Attention pre-normalization scale absorption

HYPOTHESIS: Replacing the first LayerNorm’s learned scale-and-shift with non-affine normalization plus a learned shift will reduce the model from 1,576 to 1,568 parameters while retaining at least 99% accuracy, because its scale can be absorbed exactly into the full-width QKV projection.

INTENDED_EDIT: Remove only the eight learned `ln1` scale parameters while preserving its learned shift, full eight-coordinate attention input, and all attention bias pathways.

EVIDENCE: Removing both `ln1` affine vectors together with an attention-input coordinate failed at 77.79%, so it did not isolate affine removal; the analogous MLP LayerNorm affine absorption succeeded, motivating a conservative test of only the exactly redundant attention scale.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.ln1_bias = nn.Parameter(torch.zeros(cfg.d_model))
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x) + self.ln1_bias)
>>>>>>> REPLACE