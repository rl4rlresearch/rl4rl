MECHANISM: Shared pre-sublayer normalization offset

HYPOTHESIS: Sharing the two block LayerNorm bias vectors will reduce deduplicated parameters from 1,644 to 1,636 while retaining at least 99% accuracy because both normalization sites keep learned affine transformations and all attention and MLP capacity remains intact.

INTENDED_EDIT: Tie `ln2.bias` to `ln1.bias` within the transformer block, leaving their scale parameters and all training settings unchanged.

EVIDENCE: The baseline reached 99.96%, while narrowing the MLP and deleting QKV biases caused large failures; this motivates preserving core capacity and biases while testing a minimal eight-parameter sharing constraint.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.bias = self.ln1.bias
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE