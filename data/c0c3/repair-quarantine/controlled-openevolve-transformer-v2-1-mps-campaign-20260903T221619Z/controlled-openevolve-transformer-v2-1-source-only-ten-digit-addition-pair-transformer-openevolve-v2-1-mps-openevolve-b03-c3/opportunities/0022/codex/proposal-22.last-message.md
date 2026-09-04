MECHANISM: Direct attention-bias absorption of the first LayerNorm offset

HYPOTHESIS: Making both pre-norm LayerNorms bias-free while retaining learned query and value biases will achieve at least 99% accuracy with 1,620 parameters, because the removed `ln1` offset can be absorbed into those projection biases while its induced key offset is softmax-invariant.

INTENDED_EDIT: Remove the eight learned `ln1` bias parameters and the already-qualified eight `ln2` bias parameters, preserving both LayerNorm scales, all nonlinear channels, and the critical query/value attention biases.

EVIDENCE: The `ln2`-bias-free 1,628-parameter design reached 99.95%. Removing query bias instead collapsed accuracy to 48.92%, motivating the complementary reparameterization: retain direct query/value biases and remove the redundant upstream normalization offset.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE