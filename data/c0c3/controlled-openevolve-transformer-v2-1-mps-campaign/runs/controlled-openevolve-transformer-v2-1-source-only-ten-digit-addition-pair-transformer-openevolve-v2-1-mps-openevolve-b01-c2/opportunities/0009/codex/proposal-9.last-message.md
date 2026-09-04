MECHANISM: Downstream absorption of pre-normalization scales

HYPOTHESIS: Removing both pre-sublayer LayerNorm scale vectors will reduce the model from 1,612 to 1,596 parameters while maintaining at least 99% accuracy, because each scale can be represented exactly by the following QKV or MLP input-projection weights.

INTENDED_EDIT: Make `ln1` and `ln2` entirely parameter-free while retaining their normalization behavior.

EVIDENCE: The 1,612-parameter design reached 99.95% accuracy after removing both pre-sublayer normalization biases; the remaining scales are more directly redundant because they multiply normalized features immediately before learned linear projections.

<<<<<<< SEARCH
        # Pre-sublayer normalization offsets are absorbed by downstream biases.
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        # Pre-sublayer affine transforms are absorbed by downstream projections.
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE