MECHANISM: Pre-linear LayerNorm scale absorption

HYPOTHESIS: Removing both block LayerNorm scale vectors will reduce parameters from 1,612 to 1,596 while retaining at least 99% accuracy, because each scale is immediately followed by a learned linear map that can absorb the same per-channel scaling.

INTENDED_EDIT: Make both block LayerNorms non-affine while preserving the successful `d_ff=12` architecture and training configuration.

EVIDENCE: The 1,612-parameter affine-redundancy design achieved 99.96% accuracy, while narrowing `d_ff` failed; this motivates another function-class-preserving affine reduction instead of removing hidden capacity.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE