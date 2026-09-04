MECHANISM: Single-coordinate LayerNorm scale gauge fixing

HYPOTHESIS: Fixing one `ln1` scale coordinate at one while retaining the other seven learned scales will reduce the model from 1,576 to 1,575 parameters and maintain at least 99% accuracy, because that scale can be absorbed into the corresponding QKV input-weight column and LayerNorm bias.

INTENDED_EDIT: Replace the first LayerNorm’s eight learned scales with seven learned scales plus one fixed unit scale, while retaining all eight learned shifts and the full-width attention input.

EVIDENCE: Removing all eight `ln1` scales fell to 75.48%, showing attention optimization is sensitive to wholesale scale removal; this conservative patch fixes only one analytically absorbable coordinate and preserves seven adaptive scales.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
        self.ln1_scale = nn.Parameter(torch.ones(cfg.d_model - 1))
        self.ln1_bias = nn.Parameter(torch.zeros(cfg.d_model))
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm1 = F.layer_norm(x, (x.size(-1),))
        scale1 = F.pad(self.ln1_scale, (0, 1), value=1.0)
        x = x + self.attn(norm1 * scale1 + self.ln1_bias)
        x = x + self.mlp(self.ln2(x))
        return x
>>>>>>> REPLACE